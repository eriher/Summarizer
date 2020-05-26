import copy
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, XLNetModel
from torch.nn.init import xavier_uniform_
from optimizers import Optimizer
from encoder import Classifier, ExtTransformerEncoder
#Implementation based on models used in https://github.com/nlpyang/BertSum, introduced in paper https://arxiv.org/pdf/1903.10318.pdf
#Licensed under Apache License 2.0
def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

#Will load the correct model based on input parameters.
class AutoExtSummarizer(nn.Module):
    def __init__(self, args):
        super(AutoExtSummarizer, self).__init__()
        self.args = args
        #The sentence embedding layer, named bert for reasons...
        if args.mem_length > 0:
            self.bert = AutoModel.from_pretrained(args.model_name, cache_dir=args.temp_dir, mem_len=args.mem_length)
        else:
            self.bert = AutoModel.from_pretrained(args.model_name, cache_dir=args.temp_dir)
        #The selection layer
        self.ext_layer = ExtTransformerEncoder(self.bert.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                            args.ext_dropout, args.ext_layers)
        #Load checkpoint
        if args.model_path is not '':
            checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
            if 'model' in checkpoint:
                self.load_state_dict(checkpoint['model'], strict=True)
            else:
                try:
                    self.load_state_dict(checkpoint, strict=True)
                except:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint.items():
                        name = k[7:] # remove 'module.' of dataparallel
                        new_state_dict[name]=v
                    self.load_state_dict(new_state_dict, strict=True)
        else:
            for p in self.ext_layer.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

    def forward(self, input_ids, attention_mask, clss, cls_mask, memory=None):
        #XLNet requires inverted attention mask
        if self.args.model_name.startswith('xlnet'):
            attention_mask = ~attention_mask
            if memory is not None:
                top_vec, memory = self.bert(input_ids=input_ids, input_mask=attention_mask, mems=memory)
            else:
                top_vec, memory = self.bert(input_ids=input_ids, input_mask=attention_mask)
        else:
            top_vec = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * cls_mask[:, :, None].float()
    
        sent_scores = self.ext_layer(sents_vec, cls_mask).squeeze(-1)
        return sent_scores, cls_mask, memory
   