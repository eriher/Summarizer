import torch
from torch.utils.data import Dataset
import re

#Based on Batch from Bertsum Implementation
class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None,  is_test=False, pad_token_id=0):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_labels = [x[1] for x in data]
            pre_clss = [x[2] for x in data]
            #pre_segs = [x[3] for x in data]
            src = torch.tensor(self._pad(pre_src, pad_token_id))
            labels = torch.tensor(self._pad(pre_labels, 0))
            #segs = torch.tensor(self._pad(pre_segs, pad_token_id))
            mask = ~(src == pad_token_id)

            clss = torch.tensor(self._pad(pre_clss, -1))
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0

            setattr(self, 'clss', clss)
            setattr(self, 'mask_cls', mask_cls)
            setattr(self, 'src', src)
            setattr(self, 'labels', labels)
            #setattr(self, 'segs', segs)
            setattr(self, 'mask', mask)

            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def pin_memory(self):
        self.clss = self.clss.pin_memory()
        self.src = self.src.pin_memory()
        self.mask_cls = self.mask_cls.pin_memory()
        self.labels = self.labels.pin_memory()
        self.mask = self.mask.pin_memory()
        return self

    def __len__(self):
        return self.batch_size

class CNNDMBlob(Dataset):
    def __init__(self, prefix="train", data_path="", tokenizer=None, max_pos=512, max_sentence_length=200, label_key='greedy_select'):
        pt_file = data_path + '.' + prefix + '.pt'
        self.dataset = torch.load(pt_file)
        self.tokenizer = tokenizer
        self.max_pos = max_pos
        self.max_sentence_length = max_sentence_length
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        story = ex['story']
        summary = ex['summary']
        labels = ex[self.label_key]
        tokens = [x for line in story for x in self.tokenizer.encode(line, max_length=self.max_sentence_length)]

        if len(tokens) > self.max_pos:
            if tokens[self.max_pos] == self.tokenizer.cls_token_id:
                tokens = tokens[:self.max_pos - 1]
            else:
                tokens = tokens[:self.max_pos-1] + [self.tokenizer.sep_token_id]
        clss = [index for index,token in enumerate(tokens) if token == self.tokenizer.cls_token_id]
        labels = labels[:len(clss)]

        return tokens, labels, clss, story, summary

class CNNDMBlobNoTokens(Dataset):
    def __init__(self, prefix="train", data_path="", label_key='greedy_select'):
        pt_file = data_path + '.' + prefix + '.pt'
        self.dataset = torch.load(pt_file)
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        story = ex['story']
        summary = ex['summary']
        labels = ex[self.label_key]
        return story, labels, summary

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)

def _block_tri(c, p):
    tri_c = _get_ngrams(3, rouge_clean(c.lower()).split())
    for s in p:
        tri_s = _get_ngrams(3, rouge_clean(s.lower()).split())
        if len(tri_c.intersection(tri_s)) > 0:
            return True
    return False

def rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)
    
def greedy_selection(story, summary, summary_size):

    max_rouge = 0.0
    abstract = rouge_clean(' '.join(summary)).split()
    sents = [rouge_clean(s).strip().split() for s in story]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            break
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    selected = [1 if i in selected else 0 for i in range(len(story))]
    return selected

def simple_rouge(story, summary):

    abstract = rouge_clean(' '.join(summary)).split()
    sents = rouge_clean(' '.join(story)).split()
    
    evaluated_1grams = _get_word_ngrams(1, [sents])
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = _get_word_ngrams(2, [sents])
    reference_2grams = _get_word_ngrams(2, [abstract])

    rouge_1 = cal_rouge(evaluated_1grams, reference_1grams)['f']
    rouge_2 = cal_rouge(evaluated_2grams, reference_2grams)['f']
    rouge_score = rouge_1 + rouge_2
    
    return rouge_score

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_blocks(args,src,tokenizer):
    blocks = []
    #scr text token ids
    src_ids = []
    #holds indices for cls tokens in src_ids
    cls_ids = []
    segs = []
    for i in range(len(src)):
        tokens = tokenizer.encode(src[i].lower(), max_length=200)
        if len(tokens)+len(src_ids) > args.max_pos:
            #adding another sentence would exceed limit
            blocks.append({'src': src_ids, 'clss': cls_ids, 'segs': segs})
            src_ids = []
            cls_ids = []
            segs = []
        cls_ids += [len(src_ids)]
        src_ids += tokens
        segs += [i%2] * len(tokens)
    if len(src_ids) > 0:
        blocks.append({'src': src_ids, 'clss': cls_ids, 'segs': segs})
    return blocks

def create_labeled_blocks(args,batch,tokenizer):
    src,labels,_ = batch
    blocks = []
    #scr text token ids
    src_ids = []
    #holds indices for cls tokens in src_ids
    cls_ids = []
    #segs = []
    sent_labels = []
    for i in range(len(src)):
        tokens = tokenizer.encode(src[i].lower(), max_length=200)
        if len(tokens)+len(src_ids) > args.max_pos:
            #adding another sentence would exceed limit
            blocks.append([src_ids,sent_labels,cls_ids])
            src_ids = []
            cls_ids = []
            sent_labels = []
            #segs = []
        cls_ids += [len(src_ids)]
        src_ids += tokens
        sent_labels += [labels[i]]
        #segs += [i%2] * len(tokens)
    if len(src_ids) > 0:
        blocks.append([src_ids,sent_labels,cls_ids])
    return blocks


#Clean the texts, put one sentence per line
def clean_texts(args):
    from nltk import sent_tokenize
    texts_filenames = os.listdir(args.data_dir)     
    texts = []   
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for fn in texts_filenames:
        with open(os.path.join(args.data_dir,fn),'r',encoding="utf-8",errors='ignore') as f:
            text = [unicodedata.normalize("NFKD",s.strip()+'\n') for s in sent_tokenize(f.read())]
        with open(os.path.join(args.output_dir,fn),'w',encoding="utf-8",errors='ignore') as f:
            f.writelines(text)

#Format rogue scores into format to easy copy paste into paper
def format_rouge_table(scores):
    return "& %.3f & %.3f & %.3f" % (scores['rouge-1']['f']*100, scores['rouge-2']['f']*100, scores['rouge-l']['f']*100)

#Format rogue scores into more readable format
def format_rouge_scores(scores):
    return """\n
    ** ROUGE 1
    F1        >> {:.3f}
    Precision >> {:.3f}
    Recall    >> {:.3f}
    ** ROUGE 2
    F1        >> {:.3f}
    Precision >> {:.3f}
    Recall    >> {:.3f}
    ** ROUGE L
    F1        >> {:.3f}
    Precision >> {:.3f}
    Recall    >> {:.3f}""".format(
            scores['rouge-1']['f']*100,
            scores['rouge-1']['p']*100,
            scores['rouge-1']['r']*100,
            scores['rouge-2']['f']*100,
            scores['rouge-2']['p']*100,
            scores['rouge-2']['r']*100,
            scores['rouge-l']['f']*100,
            scores['rouge-l']['p']*100,
            scores['rouge-l']['r']*100,
        )+"\n\n >> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        scores['rouge-1']['f']*100,
        scores['rouge-2']['f']*100,
        scores['rouge-l']['f']*100,
        scores['rouge-1']['r']*100,
        scores['rouge-2']['r']*100,
        scores['rouge-l']['r']*100
    )



