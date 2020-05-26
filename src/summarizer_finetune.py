# Based on examples by The HuggingFace Inc. team.
#
# Copyright 2019 The HuggingFace Inc. team.
# Copyright (c) 2019 The HuggingFace Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import functools
import logging
import os
import random
import sys
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
from transformers import (
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

from utils import (
    CNNDMBlob,
    CNNDMBlobNoTokens,
    Batch,
    create_blocks,
    create_labeled_blocks
)


from model_builder import build_optim,AutoExtSummarizer

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def collate(batch,pad_token_id=0,device=None):
    return Batch(batch, pad_token_id=pad_token_id, device=device)
def single_collate(sample):
    return sample

# ------------
# Train
# ------------

def train(args, model, tokenizer,writer):
    """ Fine-tune the pretrained model on the corpus. """
    set_seed(args)
    
    # Load the data
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = CNNDMBlob(prefix='train', data_path=args.data_dir, tokenizer=tokenizer, label_key=args.label_key)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=functools.partial(collate, pad_token_id=tokenizer.pad_token_id),
        pin_memory=True,
        num_workers=args.num_workers
    )


    # Training schedule
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = t_total // (
            len(train_dataloader) // args.gradient_accumulation_steps + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    ##Bertsum optimizer and scheduler
    if args.optim == 'bertsum':
        optimizer = build_optim(args, model, None)
    else:
        #Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total*0.1, num_training_steps=t_total)
    
    if 'score' in args.label_key:
        criterion = torch.nn.MSELoss(reduction='sum')
    else:
        criterion = torch.nn.BCELoss(reduction='sum')

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)


    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch", disable=True)

    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    start_time = time.time()
    num_docs = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")#, disable=True)

        for step, batch in enumerate(epoch_iterator):
                source, encoder_mask, target, clss, cls_mask = batch.src,batch.mask,batch.labels,batch.clss,batch.mask_cls
                num_docs += len(source)
                source = source.to(args.device)
                target = target.to(args.device)
                encoder_mask = encoder_mask.to(args.device)
                cls_mask = cls_mask.to(args.device).bool()
                clss = clss.to(args.device)

                model.train()
                outputs,mask,_ = model(
                    source,
                    encoder_mask,
                    clss,
                    cls_mask,
                )

                loss = criterion(outputs, target.float())

 
                if args.n_gpu > 1:
                   loss = loss.mean()  # mean() to average on multi-gpu parallel training
                
                #Only do this if mean loss
                #if args.gradient_accumulation_steps > 1:
                #   loss /= args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm:
                        if args.fp16:
                           torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                           torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if args.optim != 'bertsum':
                        scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if args.max_steps > 0 and global_step > args.max_steps:
                        epoch_iterator.close()
                        break
                        
                    if global_step  % args.logging_steps == 0:
                        elapsed = time.time() - start_time
                        logger.info("##STEP: %i", (global_step))
                        logger.info("Unscaled loss: %f", tr_loss)
                        logger.info('Scaled loss: %f', (tr_loss/(global_step*args.train_batch_size*args.gradient_accumulation_steps)))
                        if args.optim == 'bertsum':
                            logger.info("loss: %4.2f; lr: %7.7f; %3.0f docs/s;", (tr_loss - logging_loss)/args.logging_steps, optimizer.learning_rate,(global_step*args.train_batch_size*args.gradient_accumulation_steps)/elapsed,)
                        else:
                            logger.info("loss: %4.2f; lr: %7.7f; %3.0f docs/s;", (tr_loss - logging_loss)/args.logging_steps, scheduler.get_lr()[0],(global_step*args.train_batch_size*args.gradient_accumulation_steps)/elapsed,)
                        
                        
                        logger.info("num docs: %f",(num_docs))
                        logger.info("num docs: %f",(num_docs/elapsed))
                        if args.optim == 'bertsum':
                            writer.add_scalar("train/lr", optimizer.learning_rate, global_step)
                        else:
                            writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
                        writer.add_scalar('train/loss', (tr_loss - logging_loss)/args.logging_steps, global_step),
                        writer.add_scalar('train/loss_norm', tr_loss/(global_step*args.train_batch_size*args.gradient_accumulation_steps), global_step)
                        logging_loss = tr_loss

                    if global_step % args.eval_save_steps == 0 or global_step == 2000:
                            if not os.path.isdir(args.output_dir):
                                os.mkdir(args.output_dir)
                            checkpoint_path = os.path.join(args.output_dir, "model_step_{}.bin".format(global_step))
                            checkpoint = model.state_dict()
                            if args.n_gpu > 1:
                                from collections import OrderedDict
                                new_state_dict = OrderedDict()
                                for k, v in checkpoint.items():
                                    name = k[7:] # remove 'module.' of dataparallel
                                    new_state_dict[name]=v
                                checkpoint = new_state_dict
                            torch.save(checkpoint, checkpoint_path)


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.optim == 'bertsum':
        writer.add_scalar("train/lr", optimizer.learning_rate, global_step)
    else:
        writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
    writer.add_scalar('train/loss', (tr_loss - logging_loss)/args.logging_steps, global_step),
    writer.add_scalar('train/loss_norm', tr_loss/(global_step*args.train_batch_size*args.gradient_accumulation_steps), global_step)
    logging_loss = tr_loss     
    checkpoint_path = os.path.join(args.output_dir, "model_step_{}.bin".format(global_step))
    checkpoint = model.state_dict()
    if args.n_gpu > 1:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        checkpoint = new_state_dict
    torch.save(checkpoint, checkpoint_path)
    torch.save(args, os.path.join(args.output_dir, "training_arguments_{}.bin".format(global_step)))
    torch.save(optimizer, os.path.join(args.output_dir, "optimizer_step_{}.bin".format(global_step)))
    return global_step, tr_loss / global_step

# ------------
# Train on Full CNN/DM. Still needs work...
# ------------

def train_full(args, model, tokenizer,writer):
    """ Fine-tune the pretrained model on the corpus. """
    set_seed(args)
    
    # Load the data
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = CNNDMBlobNoTokens(prefix='train', data_path=args.data_dir, label_key=args.label_key)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=1,
        collate_fn=single_collate,
        num_workers=args.num_workers
    )


    # Training schedule
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = t_total // (
            len(train_dataloader) // args.gradient_accumulation_steps + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    ##Bertsum optimizer and scheduler
    if args.optim == 'bertsum':
        optimizer = build_optim(args, model, None)
    else:
        #Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total*0.1, num_training_steps=t_total)
    
    if 'score' in args.label_key:
        criterion = torch.nn.MSELoss(reduction='sum')
    else:
        criterion = torch.nn.BCELoss(reduction='sum')

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)


    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch", disable=True)

    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    start_time = time.time()
    num_docs = 0
    real_batch = []
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")#, disable=True)

        for step, batch in enumerate(epoch_iterator):
            num_docs += 1
            blocks = create_labeled_blocks(args,batch[0],tokenizer)
            free_slots = args.train_batch_size-len(real_batch)

            real_batch.extend(blocks[:free_slots])
            if len(real_batch) == args.train_batch_size:
                _batch = Batch(real_batch,pad_token_id=tokenizer.pad_token_id)
                source, encoder_mask, target, clss, cls_mask = _batch.src,_batch.mask,_batch.labels,_batch.clss,_batch.mask_cls
                
                source = source.to(args.device)
                target = target.to(args.device)
                encoder_mask = encoder_mask.to(args.device)
                cls_mask = cls_mask.to(args.device).bool()
                clss = clss.to(args.device)

                model.train()
                outputs,mask,_ = model(
                    source,
                    encoder_mask,
                    clss,
                    cls_mask,
                )

                #loss = criterion(outputs,target.float())
                #sumloss = loss.sum(dim=1)
                #summask = mask.float().sum(dim=1)
                #loss = (sumloss / summask).sum()
                #loss = (sumloss / summask).mean()


                loss = criterion(outputs, target.float())


                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                
                #Only do this if mean loss
                #if args.gradient_accumulation_steps > 1:
                #   loss /= args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()

                real_batch = []
                real_batch.extend(blocks[free_slots:])
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm:
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if args.optim != 'bertsum':
                        scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if args.max_steps > 0 and global_step > args.max_steps:
                        epoch_iterator.close()
                        break
                        
                    if global_step  % args.logging_steps == 0:
                        elapsed = time.time() - start_time
                        logger.info("##STEP: %i", (global_step))
                        logger.info("Unscaled loss: %f", tr_loss)
                        logger.info('Scaled loss: %f', (tr_loss/(global_step*args.train_batch_size*args.gradient_accumulation_steps)))
                        if args.optim == 'bertsum':
                            logger.info("loss: %4.2f; lr: %7.7f; %3.0f docs/s;", (tr_loss - logging_loss)/args.logging_steps, optimizer.learning_rate,(global_step*args.train_batch_size*args.gradient_accumulation_steps)/elapsed,)
                        else:
                            logger.info("loss: %4.2f; lr: %7.7f; %3.0f docs/s;", (tr_loss - logging_loss)/args.logging_steps, scheduler.get_lr()[0],(global_step*args.train_batch_size*args.gradient_accumulation_steps)/elapsed,)
                        
                        
                        logger.info("num docs: %f",(num_docs))
                        logger.info("num docs: %f",(num_docs/elapsed))
                        if args.optim == 'bertsum':
                            writer.add_scalar("train/lr", optimizer.learning_rate, global_step)
                        else:
                            writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
                        writer.add_scalar('train/loss', (tr_loss - logging_loss)/args.logging_steps, global_step),
                        writer.add_scalar('train/loss_norm', tr_loss/(global_step*args.train_batch_size*args.gradient_accumulation_steps), global_step)
                        logging_loss = tr_loss

                    if global_step % args.eval_save_steps == 0 or global_step == 2000:
                            if not os.path.isdir(args.output_dir):
                                os.mkdir(args.output_dir)
                            checkpoint_path = os.path.join(args.output_dir, "model_step_{}.bin".format(global_step))
                            checkpoint = model.state_dict()
                            if args.n_gpu > 1:
                                from collections import OrderedDict
                                new_state_dict = OrderedDict()
                                for k, v in checkpoint.items():
                                    name = k[7:] # remove 'module.' of dataparallel
                                    new_state_dict[name]=v
                                checkpoint = new_state_dict
                            torch.save(checkpoint, checkpoint_path)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.optim == 'bertsum':
        writer.add_scalar("train/lr", optimizer.learning_rate, global_step)
    else:
        writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
    writer.add_scalar('train/loss', (tr_loss - logging_loss)/args.logging_steps, global_step),
    writer.add_scalar('train/loss_norm', tr_loss/(global_step*args.train_batch_size*args.gradient_accumulation_steps), global_step)
    logging_loss = tr_loss     
    checkpoint_path = os.path.join(args.output_dir, "model_step_{}.bin".format(global_step))
    checkpoint = model.state_dict()
    if args.n_gpu > 1:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        checkpoint = new_state_dict
    torch.save(checkpoint, checkpoint_path)
    torch.save(args, os.path.join(args.output_dir, "training_arguments_{}.bin".format(global_step)))
    torch.save(optimizer, os.path.join(args.output_dir, "optimizer_step_{}.bin".format(global_step)))
    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input training data.",
    )
    parser.add_argument(
        "--temp_dir",
        default="../temp",
        type=str,
        required=False,
        help="The temp dir, used to for example store downloaded models.",
    )
    parser.add_argument(
        "--output_dir",
        default="../output",
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--log_dir",
        default="../output",
        type=str,
        help="The log directory.",
    )

    # Optional parameters
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
        help="The hugging face library model name.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--to_cpu", default=False, type=bool, help="Whether to force training on CPU."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--fp16", default=False, type=bool, help="Whether to use 16-bit floats.")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--optim", default='adamw', type=str, help="Choice of optimizer. bertsum or ")
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--beta1", default= 0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--eps", default=1e-6, type=float )
    parser.add_argument("--warmup_steps", default=10000, type=int, help="number of warmup steps")
    parser.add_argument("--max_grad_norm", default=0.0, type=float)
    parser.add_argument("--num_workers", default=0, type=int, help="number of worker processors to use for data loader")
    parser.add_argument("--model_path", default='', type=str, help="path to the model")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_save_steps', type=int, default=4000,
                        help="Save checkpoint every X updates steps.")                        
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--label_key", default='greedy_select', type=str, help="They label data key")       
    parser.add_argument("--ext_dropout", default=0.1, type=float)
    parser.add_argument("--ext_layers", default=2, type=int)
    parser.add_argument("--ext_heads", default=8, type=int)
    parser.add_argument("--ext_ff_size", default=2048, type=int)     
    parser.add_argument("--train_full", default=False, type=bool)          
    parser.add_argument("--max_pos", default=512, type=int)      
    parser.add_argument("--mem_length", default=0, type=int)
    args = parser.parse_args()



    i = 1
    output_dir = os.path.join(args.output_dir,'run%i'%i)
    while(os.path.exists(output_dir) and os.listdir(output_dir)):
        i+=1
        output_dir = os.path.join(args.output_dir,'run%i'%i)
    args.output_dir = output_dir
            

    # Set up training device
    if args.to_cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()

    # Load pretrained model and tokenizer. The decoder's weights are randomly initialized.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache=args.temp_dir)
    model = AutoExtSummarizer(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        0,
        args.device,
        args.n_gpu,
        False,
        False,
    )

    logger.info("Training/evaluation parameters %s", args)
    writer = SummaryWriter(args.output_dir)
    # Train the model
    model.to(args.device)

    if args.train_full:
        global_step, tr_loss = train_full(args, model, tokenizer,writer)
    else:
        global_step, tr_loss = train(args, model, tokenizer,writer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)



if __name__ == "__main__":
    main()
