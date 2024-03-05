# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import pandas as pd
import logging
import torch
import code
import argparse
import pickle
# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

from transformers import AutoTokenizer, AutoModel, AutoConfig

# SentEval prepare and batcher
def prepare(params, samples):
    pass

def batcher(params, batch):
    pooling = params["pooling"]
    layer = params["layer"]
    model = params["model"]
    tokenizer = params.tokenizer
    # batch = [[token for token in sent] for sent in batch]
    # batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    if "xlm" in model.config._name_or_path:
        batch = [["<s>"] + tokenizer.tokenize(sent) + ["</s>"] for sent in batch]
    else:
        batch = [["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"] for sent in batch]
    batch = [b[:512] for b in batch]
    seq_length = max([len(sent) for sent in batch])
    mask = [[1]*len(sent) + [0]*(seq_length-len(sent)) for sent in batch]
    segment_ids = [[0]*seq_length for _ in batch]
    batch = [tokenizer.convert_tokens_to_ids(sent) + [0]*(seq_length - len(sent)) for sent in batch]
    with torch.no_grad():
        batch = torch.tensor(batch).cuda()
        mask = torch.tensor(mask).cuda() # bs * seq_length
        segment_ids = torch.tensor(segment_ids).cuda()
        outputs, pooled_output, hidden_states, _ = model(batch, token_type_ids=segment_ids, attention_mask=mask, return_dict=False)

    # extended_mask = torch.cat((torch.ones(mask.shape[0], 1).cuda(), mask), -1).unsqueeze(-1)
    extended_mask = mask.unsqueeze(-1)
    if pooling == "cls":
        if layer == "all":
            output = [o.data.cpu()[:, 0].numpy() for o in hidden_states]
            embeddings = np.mean(output, 0)
        else: 
            layer = int(layer)
            output = hidden_states[layer]
            embeddings = output.data.cpu()[:, 0].numpy()
    else:
        if layer == "all":
            output = [torch.sum(extended_mask * o, -2) / torch.sum(mask, -1).unsqueeze(-1) for o in hidden_states]
            output = [o.data.cpu().numpy() for o in output]
            embeddings = np.mean(output, 0)
        else: 
            layer = int(layer)
            output = hidden_states[layer]
            output = extended_mask * output
            output = torch.sum(output, -2) / torch.sum(mask, -1).unsqueeze(-1)
            embeddings = output.data.cpu().numpy()
 
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name", default="mbert", type=str,
                        choices=["bert", "mbert", "xlmr"],
                        help="the name of transformer model to evaluate on")
    parser.add_argument("--task_index", default=None, type=int,
                        help="which task to perform for original senteval English tasks")
    parser.add_argument("--language", default=None, type=str,
                        choices=["Arabic", "Chinese", "Hebrew", "Hindi", "Russian", "Tamil", "Korean", "Japanese",
                                 "English", "English_UD", "Coptic", "Sanskrit",
                                 "Xru", "Xde", "Xes", "Xfi", "Xfr", "Xtr" ])
    parser.add_argument("--pooling", default="cls", type=str,
                        choices=["cls", "mean"],
                        help="which layer to evaluate on")
    parser.add_argument("--layer", default="all", type=str,
                        help="which layer to evaluate on")
    parser.add_argument("--seed", default=1111, type=int,
                        help="which seed to use")
    args = parser.parse_args()

    model_dict = {"bert": "bert-base-cased",
                  "mbert": "bert-base-multilingual-cased",
                  "xlmr": "xlm-roberta-base",}

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    config = AutoConfig.from_pretrained(model_dict[args.model_name])
    config.output_hidden_states = True
    config.output_attentions = True
    tokenizer = AutoTokenizer.from_pretrained(model_dict[args.model_name])
    model = AutoModel.from_pretrained(model_dict[args.model_name], config=config).cuda()
    model.eval()

    # Set params for DiscoEval or SentEval
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16,
              'tokenizer': tokenizer, "pooling": args.pooling, "layer": args.layer, "model": model, 'seed': args.seed}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}

    se = senteval.engine.SE(params, batcher, prepare)
    transfer_tasks_senteval = [
        ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC'], # stand-alone sentence classification
        ['MRPC', 'SNLI', 'SICKEntailment'], # pair-sentence clasificationc
        ['SICKRelatedness', 'STSBenchmark'], # supervised semantic similarity
        ['STS12', 'STS13', 'STS14', 'STS15', 'STS16'], # unsupervised semantic similarity
        ['Length', 'WordContent', 'Depth', 'TopConstituents',
         'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
         'OddManOut', 'CoordinationInversion'], # probing tasks
    ]

    transfer_tasks = [
        ["Ar_SubjNumber", "Ar_ObjNumber", "Ar_Aspect", "Ar_Mood", "Ar_Voice", "Ar_PronType", "Ar_SubjDefinite",
         "Ar_ObjDefinite"],
        ["Zh_Aspect", "Zh_Voice"],
        ["He_Tense", "He_SubjNumber", "He_ObjNumber", "He_Voice", "He_VerbForm", "He_PronType", "He_SubjGender",
         "He_ObjGender", "He_SubjDefinite", "He_ObjDefinite"],
        ["Hi_Tense", "Hi_SubjNumber", "Hi_ObjNumber", "Hi_Aspect", "Hi_Mood", "Hi_Voice", "Hi_VerbForm", "Hi_PronType",
         "Hi_SubjGender", "Hi_ObjGender"],
        ["Ru_Tense", "Ru_SubjNumber", "Ru_ObjNumber", "Ru_Aspect", "Ru_Mood", "Ru_Voice", "Ru_VerbForm",
         "Ru_SubjGender", "Ru_ObjGender"],
        ["Ta_Tense", "Ta_SubjNumber", "Ta_ObjNumber", "Ta_Mood", "Ta_VerbForm"],
        ["Cop_PronType", "Cop_SubjDefinite", "CopObjDefinite"],
        ["Sa_Tense", "Sa_SubjNumber", "Sa_ObjNumber", "Sa_Mood", "Sa_VerbForm", "Sa_SubjGender", "Sa_ObjGender"],
        ["En_Tense", "En_SubjNumber", "En_ObjNumber", "En_Mood", "En_VerbForm", "En_PronType"],
        ['Xde_Length', 'Xde_WordContent', 'Xde_Depth',
         'Xde_BigramShift', 'Xde_Tense', 'Xde_SubjNumber', 'Xde_ObjNumber',
         'Xde_OddManOut', 'Xde_CoordinationInversion'],
        ['Xes_Length', 'Xes_WordContent', 'Xes_Depth',
         'Xes_BigramShift', 'Xes_Tense', 'Xes_SubjNumber', 'Xes_ObjNumber',
         'Xes_OddManOut', 'Xes_CoordinationInversion'],
        ['Xfi_Length', 'Xfi_WordContent', 'Xfi_Depth',
         'Xfi_BigramShift', 'Xfi_Tense', 'Xfi_SubjNumber', 'Xfi_ObjNumber',
         'Xfi_OddManOut', 'Xfi_CoordinationInversion'],
        ['Xfr_Length', 'Xfr_WordContent', 'Xfr_Depth',
         'Xfr_BigramShift', 'Xfr_Tense', 'Xfr_SubjNumber', 'Xfr_ObjNumber',
         'Xfr_OddManOut', 'Xfr_CoordinationInversion'],
        ['Xru_Length', 'Xru_WordContent', 'Xru_Depth',
         'Xru_BigramShift', 'Xru_Tense', 'Xru_SubjNumber', 'Xru_ObjNumber',
         'Xru_OddManOut', 'Xru_CoordinationInversion'],
        ['Xtr_Length', 'Xtr_WordContent', 'Xtr_Depth',
         'Xtr_BigramShift', 'Xtr_Tense', 'Xtr_SubjNumber', 'Xtr_ObjNumber',
         'Xtr_OddManOut', 'Xtr_CoordinationInversion']
    ]
    if args.language == "Arabic":
        results = se.eval(transfer_tasks[0])
    elif args.language == "Chinese":
        results = se.eval(transfer_tasks[1])
    elif args.language == "Hebrew":
        results = se.eval(transfer_tasks[2])
    elif args.language == "Hindi":
        results = se.eval(transfer_tasks[3])
    elif args.language == "Russian":
        results = se.eval(transfer_tasks[4])
    elif args.language == "Tamil":
        results = se.eval(transfer_tasks[5])
    elif args.language == "Coptic":
        results = se.eval(transfer_tasks[6])
    elif args.language == "Sanskrit":
        results = se.eval(transfer_tasks[7])
    elif args.language == "English_UD":
        results = se.eval(transfer_tasks[8])
    elif args.language == "Xde":
        results = se.eval(transfer_tasks[9])
    elif args.language == "Xes":
        results = se.eval(transfer_tasks[10])
    elif args.language == "Xfi":
        results = se.eval(transfer_tasks[11])
    elif args.language == "Xfr":
        results = se.eval(transfer_tasks[12])
    elif args.language == "Xru":
        results = se.eval(transfer_tasks[13])
    elif args.language == "Xtr":
        results = se.eval(transfer_tasks[14])
    elif args.language == "English" or None:
        assert args.task_index is not None
        results = se.eval(transfer_tasks_senteval[args.task_index])
    #print(results)
    if args.language != None and args.task_index == None:
        output_path = '{}_p={}_l={}_lg={}_s={}'.format(
            args.model_name,
            args.pooling,
            args.layer,
            args.language,
            params['seed'])

        pred_path = '{}_p={}_l={}_lg={}_s={}_preds'.format(
            args.model_name,
            args.pooling,
            args.layer,
            args.language,
            params['seed'])

    else:
        output_path = '{}_p={}_l={}_t={}_s={}'.format(
            args.model_name,
            args.pooling,
            args.layer,
            args.task_index,
            params['seed'])

        pred_path = '{}_p={}_l={}_t={}_s={}_preds'.format(
            args.model_name,
            args.pooling,
            args.layer,
            args.task_index,
            params['seed'])

    with open(output_path + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



    # scores = {}
    # predictions = {}
    # for key, value in results.items():
    #     keydict_score = {}
    #     metric_dict = {}
    #     if args.task_index == 3:
    #         for dataset, val in value.items():
    #             for metric, valu in val.items():
    #                 if metric == 'predictions':
    #                     predictions[(key, dataset)] = valu
    #                 else:
    #                     metric_dict[metric] = valu
    #             keydict_score[dataset] = metric_dict
    #         scores[key] = keydict_score
    #     else:
    #         for k, v in value.items():
    #             if k == 'predictions':
    #                 predictions[key] = v
    #             else:
    #                 keydict_score[k] = v
    #         scores[key] = keydict_score
    #
    # df_scores = pd.DataFrame(scores)
    # df_preds = pd.DataFrame(predictions)
    # df_scores.to_csv(output_path, index=True)
    # df_preds.to_csv(pred_path, index=True)
