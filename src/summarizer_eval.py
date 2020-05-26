import os
import functools
import torch
import numpy as np
from model_builder import AutoExtSummarizer
import time
import argparse
import pyrouge
import shutil
import tempfile
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from utils import (
    CNNDMBlob,
    CNNDMBlobNoTokens,
    Batch,
    simple_rouge,
    str2bool,
    _block_tri,
    create_blocks,
    create_labeled_blocks,
    format_rouge_table,
    format_rouge_scores
)
import sys
from tqdm import tqdm, trange
from rouge import Rouge
import re
from tensorboardX import SummaryWriter
from average_checkpoints import average_checkpoints
import unicodedata
import math

# ---------------------
# Utils
# ---------------------

def collate(batch,pad_token_id=0,device=None, is_test=False):
    return Batch(batch, pad_token_id=pad_token_id, device=device, is_test=is_test)

def single_collate(sample):
    return sample

#Calculates sentence similarity score, using sentence bert average for a "document" embedding
def calc_sbert_similarity(pred,gold):
    from sentence_transformers import SentenceTransformer
    import scipy
    embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    assert(len(pred) == len(gold))
    scores = []

    for p,g in zip(pred,gold):
        candidate = np.mean(embedder.encode(p),0)
        reference = np.mean(embedder.encode(g),0)
        scores.append((scipy.spatial.distance.cdist([candidate], [reference], "cosine")[0]))
    return 100-(np.mean(scores)*100)
    #return (1/(1 + np.exp(-np.mean(scores))))*100 

#Calculates the sentence coverage score. Assumes one sentence per line.
def calc_sentence_coverage(cand,ref):
    scores = []
    for c,r in zip(cand,ref):
        candidate_set = set(np.array(c))
        reference_set = set(np.array(r))
        overlap = len(candidate_set.intersection(reference_set))/len(reference_set.union(candidate_set))
        scores.append(overlap)
    return np.mean(scores)*100


# ---------------------
# Baseline Generators
# ---------------------

#Calculates reference scores, for every-7
def cnndm_every7(args,logger):
    eval_dataset = CNNDMBlobNoTokens(prefix='test', data_path=args.data_dir)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=lambda x: x, num_workers=args.num_workers
    )
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))

    gold = []
    every7 = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        stories = [x[0] for x in batch]
        summaries = [x[2] for x in batch]
        
        for story,summary in zip(stories,summaries):
            every7.append([s for i,s in enumerate(story) if i % 7 == 0][:3])
            gold.append(summary)
    rouge = Rouge()


    simple_rouge_score_greed = np.mean([simple_rouge(every7[i],gold[i]) for i in range(len(gold))])*100
    rouge_score_greed = format_rouge_scores(rouge.get_scores([" ".join(p) for p in every7], [" ".join(g) for g in gold], avg=True))
    similarity_score_greed = calc_sbert_similarity(every7,gold)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    rouge_output_file = os.path.join(args.output_dir,"cnndm_every7.txt")
    with open(rouge_output_file, 'w', encoding="utf-8") as f:
        f.write("Every-7")
        f.write(rouge_score_greed)
        f.write("Similarity score(sbert): %f\n" % similarity_score_greed)

#Calculates reference rouge scores, for lead-n and the labels
def cnndm_lead_n(args,logger):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eval_dataset = CNNDMBlobNoTokens(prefix='test', data_path=args.data_dir)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=lambda x: x, num_workers=args.num_workers
    )
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))

    gold = []
    pred = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        stories = [x[0] for x in batch]
        summaries = [x[1] for x in batch]
        
        for story,summary in zip(stories,summaries):
            pred.append(story[:3])
            gold.append(summary)
    rouge = Rouge()
    simple_rouge_score_greed = np.mean([simple_rouge(every7[i],gold[i]) for i in range(len(gold))])*100
    rouge_score_greed = format_rouge_scores(rouge.get_scores([" ".join(p) for p in every7], [" ".join(g) for g in gold], avg=True))
    similarity_score_greed = calc_sbert_similarity(every7,gold)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    rouge_output_file = os.path.join(args.output_dir,"cnndm_every7.txt")
    with open(rouge_output_file, 'w', encoding="utf-8") as f:
        f.write("lead-3")
        f.write(rouge_score_greed)
        f.write("\nSimple rouge score: %f\n" % simple_rouge_score_greed)
        f.write("Similarity score(sbert): %f\n" % similarity_score_greed)


#Generate Baseline by selecting every-7th sentence.
def gen_baseline(args):
    files1 = os.listdir(args.data_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for fn in files1:
        with open(os.path.join(args.data_dir,fn),'r',encoding="utf-8",errors='ignore') as f:
            text = [s.strip()+'\n' for i,s in enumerate(f.readlines()) if i % 7 == 0]
        with open(os.path.join(args.output_dir,fn),'w',encoding="utf-8",errors='ignore') as f:
            f.writelines(text)

#Calculates reference rouge scores, for the labels
def cnndm_label_ref(args,logger):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eval_dataset = CNNDMBlobNoTokens(prefix='test', data_path=args.data_dir, tokenizer=tokenizer, label_key=args.label_key, max_pos=9999)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=functools.partial(collate, pad_token_id=tokenizer.pad_token_id, is_test=True), num_workers=args.num_workers
    )
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))

    gold = []
    lead3pred = []
    greedyselectpred = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        source = batch.src
        encoder_mask = batch.mask
        scores = batch.labels
        clss = batch.clss
        cls_mask = batch.mask_cls

        scores = np.argsort(-scores, 1)
        
        
        for idx,row in enumerate(scores):
            _pred = []

            for jdx,i in enumerate(row):
                if batch.labels[idx][i] == 0:
                    break
                
                if i >= len(batch.src_str[idx]):
                    continue
                candidate = batch.src_str[idx][i].strip()

                _pred.append(candidate)
                if len(_pred) == 3:
                   break

            greedyselectpred.append(_pred)
            lead3pred.append(batch.src_str[idx][:3])
            gold.append(batch.tgt_str[idx])

    rouge = Rouge()
    simple_rouge_score_greed = np.mean([simple_rouge(greedyselectpred[i],gold[i]) for i in range(len(gold))])*100
    rouge_score_greed = format_rouge_scores(rouge.get_scores([" ".join(p) for p in greedyselectpred], [" ".join(g) for g in gold], avg=True))
    similarity_score_greed = calc_sbert_similarity(greedyselectpred,gold)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    rouge_output_file = os.path.join(args.output_dir,"cnndm_ref_results.txt")
    with open(rouge_output_file, 'w', encoding="utf-8") as f:
        f.write("GREED")
        f.write(rouge_score_greed)
        f.write("\nSimple rouge score: %f\n" % simple_rouge_score_greed)
        f.write("Similarity score(sbert): %f\n" % similarity_score_greed)

# ----------------------------
# CNN/DM Checkpoint Validation
# ----------------------------

#Performs validation on cnn/dm "validation" on all the checkpoints in model_path folder and returns a list sorted by descending score. Used for creating the average checkpoint
def cnndm_validation(args):
    #read all models in directory
    models = [os.path.join(args.model_path, model) for model in os.listdir(args.model_path) if "model" in model]
    models = sorted(models, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
    #add initial empty checkpoint
    models.insert(0,'')
    
    writer = SummaryWriter(args.output_dir, filename_suffix="valid")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eval_dataset = CNNDMBlob(prefix='valid', data_path=args.data_dir, tokenizer=tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=functools.partial(collate, pad_token_id=tokenizer.pad_token_id, is_test=True), num_workers=args.num_workers, pin_memory=True
    )
    results = []
    #for each model run evaluation and calculate simple rouge score and loss
    for m in models:
        args.model_path = m
        model = AutoExtSummarizer(args)
        model.to(args.device)
        model.eval()
        if "score" in args.label_key:
            criterion = torch.nn.BCELoss(reduction='sum')
        else:
            criterion = torch.nn.MSELoss(reduction='sum')
        try:
            step = [int(s) for s in os.path.basename(m).split('.')[0].split('_') if s.isdigit()][0]
        except:
            step = 0
        eval_loss = 0.0
        nb_eval_steps = 0
        
        gold = []
        pred = []
        avg_loss = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):#, disable=True):
            source = batch.src.to(args.device)
            encoder_mask = batch.mask.to(args.device)
            scores = batch.labels.to(args.device)
            clss = batch.clss.to(args.device)
            cls_mask = batch.mask_cls.to(args.device).bool()
            target = batch.labels.to(args.device)

            with torch.no_grad():
                sent_scores,mask,memory = model(
                    source,
                    encoder_mask,
                    clss,
                    cls_mask,
                )
                loss = criterion(sent_scores, target.float())
                avg_loss += loss.item()
                #Seperates padding from the ones that are actually 0
                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()
                #Returns indexes sorted by descending score
                selected_ids = np.argsort(-sent_scores, 1)
                
                for idx,row in enumerate(selected_ids):
                    _pred = []
                    for i in row:
                        if i >= len(batch.src_str[idx]):
                            continue
                        candidate = batch.src_str[idx][i].strip()
                        if (not _block_tri(candidate, _pred)):
                            _pred.append(candidate)
                        if len(_pred) == 3:
                            break
                    pred.append(_pred)
                        
                    _gold = batch.tgt_str[idx]
                    gold.append(_gold)
                #break

        score = []
        for i in range(len(gold)):
            score.append(simple_rouge(pred[i],gold[i]))
        avg_loss = avg_loss/(len(eval_dataloader)*args.batch_size)
        score = np.mean(score)*100
        print("Simple rouge score(f1+f2): %f" % score)
        print("Loss: %f" % avg_loss)
        writer.add_scalar('validation/score', score, step)
        writer.add_scalar('validation/loss', avg_loss, step)
        results.append(score-avg_loss)
    writer.close()
    top_scores = np.argsort(-np.array(results))[:3]

    return [m for i,m in enumerate(models) if i in top_scores]

# ----------------------------
# CNN/DM Evaluation
# ----------------------------

#Tests the model on the cnn/dm "test" dataset
def cnndm_test(args,model,logger):
    #Init model
    model = AutoExtSummarizer(args)
    model.to(args.device)
    model.eval()

    if "score" in args.label_key:
        criterion = torch.nn.BCELoss(reduction='sum')
    else:
        criterion = torch.nn.MSELoss(reduction='sum')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eval_dataset = CNNDMBlob(prefix='test', data_path=args.data_dir, tokenizer=tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=functools.partial(collate, pad_token_id=tokenizer.pad_token_id, is_test=True), num_workers=args.num_workers, pin_memory=True
    )

    logger.info("***** Running CNNDM evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    eval_loss = 0.0
    nb_eval_steps = 0

    gold = []
    pred = []
    avg_loss = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):#, disable=True):
        source = batch.src.to(args.device)
        encoder_mask = batch.mask.to(args.device)
        scores = batch.labels.to(args.device)
        clss = batch.clss.to(args.device)
        cls_mask = batch.mask_cls.to(args.device).bool()
        target = batch.labels.to(args.device)

        with torch.no_grad():
            sent_scores,mask,memory = model(
                source,
                encoder_mask,
                clss,
                cls_mask,
            )
            loss = criterion(sent_scores, target.float())
            avg_loss += loss.item()
            #Seperates padding from the ones that are actually 0
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()
            #Returns indexes sorted by descending score
            selected_ids = np.argsort(-sent_scores, 1)
            
            for idx,row in enumerate(selected_ids):
                _pred = []
                for i in row:
                    if i >= len(batch.src_str[idx]):
                        continue
                    candidate = batch.src_str[idx][i].strip()
                    if args.block_tri:
                        if (not _block_tri(candidate, _pred)):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)
                    if len(_pred) == 3:
                        break
                pred.append(_pred)
                _gold = batch.tgt_str[idx]
                gold.append(_gold)
            #break
    simple_rouge_score = []
    for i in range(len(gold)):
        simple_rouge_score.append(simple_rouge(pred[i],gold[i]))
    simple_rouge_score = np.mean(simple_rouge_score)*100
    avg_loss = avg_loss/(len(eval_dataloader)*args.batch_size)

    #python rouge implementation
    rouge = Rouge()
    rouge_score = rouge.get_scores([" ".join(p) for p in pred], [" ".join(g) for g in gold], avg=True)
    rouge_score_formatted = format_rouge_scores(rouge_score)
    rouge_table = format_rouge_table(rouge_score)

    similarity_score = calc_sbert_similarity(pred,gold)

    print(rouge_score_formatted)
    print("Simple rouge score: %.3f" % simple_rouge_score)
    print("Avg loss: %.3f" % avg_loss)
    print("Similarity score(sbert): %.3f" % similarity_score)
    print(rouge_table + " & %.3f & %.3f" % (simple_rouge_score, similarity_score))


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    rouge_output_file = os.path.join(args.output_dir,"cnndm_test_results_{}_{}.txt".format(args.model_name,os.path.basename(args.model_path).split(".")[0]))
    with open(rouge_output_file, 'w', encoding="utf-8") as f:
        f.write(rouge_score_formatted)
        f.write("\nSimple rouge score: %.3f\n" % simple_rouge_score)
        f.write("Avg loss: %.3f\n" % avg_loss)
        f.write("Similarity score(sbert): %.3f\n" % similarity_score)
        f.write(rouge_table + " & %.3f & %.3f" % (simple_rouge_score, similarity_score))

#Perform test using the "full" text instead of just max 512 block size
def cnndm_test_full(args,model,logger):
    model = AutoExtSummarizer(args)
    model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = CNNDMBlobNoTokens(prefix='test', data_path=args.data_dir, label_key=args.label_key)
    train_sampler = SequentialSampler(train_dataset)
    model_collate_fn = functools.partial(collate, pad_token_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=1, collate_fn=single_collate, num_workers=args.num_workers
    )
    

    logger.info("***** Running CNNDM evaluation  *****")
    logger.info("  Num examples = %d", len(train_dataset))
    
    gold = []
    pred = []

    for batch in tqdm(train_dataloader, desc="Evaluating"):#, disable=True):

        summary = batch[0][2]
        story = batch[0][0]

        blocks = create_labeled_blocks(args,batch[0],tokenizer)
        block_scores = []
        memory = None
        for block in blocks:
            _batch = Batch([block],pad_token_id=tokenizer.pad_token_id)
            source = _batch.src.to(args.device)
            encoder_mask = _batch.mask.to(args.device)
            clss = _batch.clss.to(args.device)
            cls_mask = _batch.mask_cls.to(args.device).bool()

            with torch.no_grad():
                sent_scores,mask,memory = model(
                    source,
                    encoder_mask,
                    clss,
                    cls_mask,
                    memory = memory
                )
                #Seperates padding from the ones that are actually 0
                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()
                block_scores.extend(sent_scores[0])
        selected_ids = np.argsort(block_scores)[::-1]
        _pred = []
        for i in selected_ids:
            candidate = story[i].strip()
            if (not _block_tri(candidate, _pred)):
                _pred.append(candidate)            
            if len(_pred) == 3:
                break
        pred.append(_pred)
        gold.append(summary)

    #python rouge implementation
    rouge = Rouge()
    rouge_score = rouge.get_scores([" ".join(p) for p in pred], [" ".join(g) for g in gold], avg=True)
    rouge_score_formatted = format_rouge_scores(rouge_score)
    rouge_table = format_rouge_table(rouge_score)

    similarity_score = calc_sbert_similarity(pred,gold)
    #similarity_score = 0

    print(rouge_score_formatted)
    print("Similarity score(sbert): %.3f" % similarity_score)
    print(rouge_table + " & %.3f" % similarity_score)


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    rouge_output_file = os.path.join(args.output_dir,"cnndm_test_full_results_{}_{}.txt".format(args.model_name,os.path.basename(args.model_path).split(".")[0]))
    with open(rouge_output_file, 'w', encoding="utf-8") as f:
        f.write(rouge_score_formatted)
        f.write("Similarity score(sbert): %.3f\n" % similarity_score)
        f.write(rouge_table + " & %.3f" % (similarity_score))        

# ----------------------------
# Summary Generation
# ----------------------------

def generate_summaries(args):
    #init model..
    model = AutoExtSummarizer(args)
    model.to(args.device)
    model.eval()
    #Read texts
    texts_filenames = os.listdir(args.data_dir)     
    texts = []   
    for fn in texts_filenames:
        with open(os.path.join(args.data_dir,fn),'r',encoding="utf-8",errors='ignore') as f:
            text = f.readlines()
            texts.append(text)
    #Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    results = []
    with torch.no_grad():
        for text in texts:
            blocks = create_blocks(args,text,tokenizer)
            block_scores = []
            memory = None

            for block in blocks:
                src = block['src']
                clss = block['clss']

                mask = [True]*len(src)
                cls_mask = [True]*len(clss)
                source = torch.tensor([src]).to(args.device)
                
                encoder_mask = torch.tensor([mask]).to(args.device)
                clss = torch.tensor([clss]).to(args.device)
                cls_mask = torch.tensor([cls_mask]).to(args.device)
                
                sent_scores,mask,memory = model(
                    source,
                    encoder_mask,
                    clss,
                    cls_mask,
                    memory)


                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores[0].cpu().data.numpy()

                if len(block['clss']) == 1:
                    sent_scores = [sent_scores]
                block_scores.extend(sent_scores)
            sorted_idxs = np.argsort(block_scores)[::-1]
            _pred = []
            selected = []
            for ind in sorted_idxs:
                candidate = text[ind].strip()
                if args.block_tri:
                    if (not _block_tri(candidate, _pred)):
                        _pred.append(candidate)
                        selected.append(ind)
                else:
                    _pred.append(candidate)
                    selected.append(ind)
                if len(selected) == math.ceil(len(text)*(1/7)):
                    break            
            results.append([text[i].strip()+'\n' for i in sorted(selected)])
    if not os.path.exists(args.cand_dir):
        os.makedirs(args.cand_dir)
    for fn,text in zip(texts_filenames,results):
        with open(os.path.join(args.cand_dir,fn),'w',encoding="utf-8") as f:
            f.writelines(text)

# ----------------------------
# Summary Evaluation
# ----------------------------

#Runs evaluation on two folders containing candidate and reference summaries. Assumes one sentence per line.
def eval_texts_folders(args):
    files1 = os.listdir(args.ref_dir)
    files2 = os.listdir(args.cand_dir)
    assert(files1 == files2)
    ref = []
    cand = []
    for fn in files1:
        with open(os.path.join(args.ref_dir,fn),'r',encoding="utf-8",errors='ignore') as f:
            ref.append([line.strip() for line in f.readlines()])
        with open(os.path.join(args.cand_dir,fn),'r',encoding="utf-8",errors='ignore') as f:
            cand.append([line.strip() for line in f.readlines()])

    
    
    rouge = Rouge()
    rouge_score = rouge.get_scores([" ".join(p) for p in cand], [" ".join(g) for g in ref], avg=True)
    formatted_rouge_score = format_rouge_scores(rouge_score)
    rouge_table = format_rouge_table(rouge_score)
    similarity_score = calc_sbert_similarity(ref,cand)
    sentence_overlap = calc_sentence_coverage(cand,ref)
    print(formatted_rouge_score)
    print("Similarity score(sbert): %.3f" % similarity_score)
    print("Sentence overlap: %.3f" % sentence_overlap)
    print(rouge_table + " & %.3f & %.3f" % (similarity_score,sentence_overlap))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    rouge_output_file = os.path.join(args.output_dir,"text_eval_results_{}_{}.txt".format(os.path.basename(args.cand_dir),os.path.basename(args.ref_dir)))
    with open(rouge_output_file, 'w', encoding="utf-8") as f:
        f.write(formatted_rouge_score)
        f.write("Sentence overlap: %.3f\n" % sentence_overlap)
        f.write("Similarity score(sbert): %.3f\n" % similarity_score)
        f.write(rouge_table + " & %.3f & %.3f \\\\" % (similarity_score, sentence_overlap))
   

# ----------------------------
# Confidence Metrics
# ----------------------------

def texts_conf(args):
    #requires model..
    model = AutoExtSummarizer(args)
    model.to(args.device)
    #Read texts
    texts_filenames = os.listdir(args.data_dir)     
    texts = []   
    for fn in texts_filenames:
        with open(os.path.join(args.data_dir,fn),'r',encoding="utf-8",errors='ignore') as f:
            text = f.readlines()
            texts.append(text)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    results = []
    sorted_results = []
    with torch.no_grad():
        for text in texts:
            blocks = create_blocks(args,text,tokenizer)
            block_scores = []
            memory = None

            for block in blocks:
                src = block['src']
                clss = block['clss']

                mask = [True]*len(src)
                cls_mask = [True]*len(clss)
                source = torch.tensor([src]).to(args.device)
                
                encoder_mask = torch.tensor([mask]).to(args.device)
                clss = torch.tensor([clss]).to(args.device)
                cls_mask = torch.tensor([cls_mask]).to(args.device)
                
                sent_scores,mask,memory = model(
                    source,
                    encoder_mask,
                    clss,
                    cls_mask,)

                sent_scores = sent_scores[0].cpu().data.numpy()
                if len(block['clss']) == 1:
                    sent_scores = [sent_scores]
                block_scores.extend(sent_scores)
            if len(block_scores) < 100:
               continue
            results.append(block_scores)
            sorted_results.append(sorted(block_scores)[::-1])

    length = len(results[0])
    for res in results:
        if length > len(res):
            length = len(res)
    
    results = [res[:length] for res in results]
    results = np.mean(results,0)
    sorted_results = [res[:length] for res in sorted_results]
    sorted_results = np.mean(sorted_results,0)
    
    write_conf(args,'texts','conf',sorted_results)
    write_conf(args,'texts','pos',results)

def cnndm_conf(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eval_dataset = CNNDMBlob(prefix='test', data_path=args.data_dir, tokenizer=tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,
        collate_fn=functools.partial(collate, pad_token_id=tokenizer.pad_token_id),
        pin_memory=True,
        num_workers=args.num_workers
    )
    model = AutoExtSummarizer(args)
    model.to(args.device)
    model.eval()

    results = []
    sorted_results = []
    for step, batch in enumerate(tqdm(eval_dataloader,"iterator")):

        source = batch.src.to(args.device)
        encoder_mask = batch.mask.to(args.device)
        scores = batch.labels.to(args.device)
        clss = batch.clss.to(args.device)
        cls_mask = batch.mask_cls.to(args.device).bool()
        target = batch.labels.to(args.device)
        if target.size()[1] < 18:
            continue
        with torch.no_grad():
            sent_scores,mask,memory = model(
                source,
                encoder_mask,
                clss,
                cls_mask,
            )

            #Seperates padding from the ones that are actually 0
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()[0]
            results.append(sent_scores)
            sorted_results.append(sorted(sent_scores)[::-1])
    length = len(results[0])
    for arr in results:
        if len(arr) < length:
            length = len(arr)
    
    results = [a[:length] for a in results]
    results = np.mean(results,0)

    sorted_results = [a[:length] for a in sorted_results]
    sorted_results = np.mean(sorted_results,0)    

    write_conf(args,'cnndm','conf',sorted_results)
    write_conf(args,'cnndm','pos',results)
    return

def write_conf(args,ds,ps,results):
    selected = math.ceil(len(results)/7)
    with open(os.path.join(args.output_dir,"{}_{}_{}.txt".format(args.model_name,ds,ps)), 'w', encoding="utf-8") as f:
        f.write("length: %s\n" % len(results))
        f.write("Max: %s\n" % max(results))    
        f.write("Min: %s\n" % min(results))

        # Printing the mean 
        f.write("Mean is :%s\n" % np.mean(results))     
        # Printing the mean 
        f.write("Std is :%s\n" % np.std(results))   
        f.write("Min Selected: %s\n" % min(results[:selected]))

        # Printing the mean 
        f.write("Mean Selected is :%s\n" % np.mean(results[:selected]))     
        # Printing the mean 
        f.write("Std Selected is :%s\n" % np.std(results[:selected]))             
    with open(os.path.join(args.output_dir,"{}_{}_{}.csv".format(args.model_name,ds,ps)), 'w') as f:
        f.write("num, score\n")
        for i,score in enumerate(results):
            f.write("%i, %f\n" % (i,score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='', type=str)
    parser.add_argument("--output_dir", default='../output', type=str)
    parser.add_argument("--ext_dropout", default=0.1, type=float)
    parser.add_argument("--ext_layers", default=2, type=int)
    parser.add_argument("--ext_heads", default=8, type=int)
    parser.add_argument("--ext_ff_size", default=2048, type=int)
    parser.add_argument("--ext_layer", default='transformer', type=str, choices=['classifier', 'transformer', 'k_means'])
    parser.add_argument("--temp_dir", default='../temp')
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_pos", default=512, type=int)
    parser.add_argument("--batch_size", default=36, type=int)
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
        help="The model checkpoint to initialize the encoder and decoder's weights with.",
    )
    parser.add_argument(
        "--to_cpu", default=False, type=bool, help="Whether to force training on CPU."
    )
    parser.add_argument(
        "--num_workers", default=0, type=int,
    )
    parser.add_argument("--min_words_per_sent", default=3, type=int)
    parser.add_argument("--ref", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--validate", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--test", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--test_full", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--ref_dir", default='', type=str)
    parser.add_argument("--cand_dir", default='', type=str)
    parser.add_argument("--test_texts", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--gen", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--gen_baseline", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--block_tri", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--label_key", default='greedy_select', type=str)
    parser.add_argument("--every7", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--texts_conf", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--cnndm_conf", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--mem_length", default=0, type=int)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.to_cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()


    if args.validate:
        path = os.path.join(args.output_dir, "model_avg.bin")
        torch.save(average_checkpoints(cnndm_validation(args)), path)
        args.model_path = path
    if args.test:
        cnndm_test(args,model,logger)
    if args.test_full:
        cnndm_test_full(args,model,logger)        
    if args.gen:
        generate_summaries(args)
    if args.gen_baseline:
        gen_baseline(args)
    if args.test_texts:
        eval_texts_folders(args)
    if args.label_ref:
        cnndm_label_ref(args,logger)
    if args.every7:
        cnndm_every7(args,logger)
    if args.texts_conf:
        texts_conf(args)
    if args.cnndm_conf:
        cnndm_conf(args)
    if args.xlnet_conf:
        xlnet_conf(args)
  

if __name__ == "__main__":
    main()