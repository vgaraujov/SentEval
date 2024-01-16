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
    batch = [[token for token in sent] for sent in batch]
    batch = [" ".join(sent) if sent != [] else "." for sent in batch]
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
                        choices=["bert", "mbert"],
                        help="the name of transformer model to evaluate on")
    parser.add_argument("--task_index", default=None, type=int,
                        help="which task to perform for original senteval English tasks")
    parser.add_argument("--language", default=None, type=str,
                        choices=["Arabic", "Chinese", "Hebrew", "Hindi", "Russian", "Tamil", "Korean", "Japanese",
                                 "English"])
    parser.add_argument("--pooling", default="cls", type=str,
                        choices=["cls", "mean"],
                        help="which layer to evaluate on")
    parser.add_argument("--layer", default="all", type=str,
                        help="which layer to evaluate on")
    parser.add_argument("--seed", default=1111, type=int,
                        help="which seed to use")
    args = parser.parse_args()

    model_dict = {"bert": "bert-base-cased",
                  "mbert": "bert-base-multilingual-cased"}

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
        # ['Mr_Aspect', 'Mr_Case', 'Mr_Deixis', 'Mr_Gender', 'Mr_Number', 'Mr_Person', 'Mr_Polarity',
        # 'Mr_PronType', 'Mr_Tense', 'Mr_VerbForm']  # Marathi probing tasks
    ]

    transfer_tasks = [
        ["Ar_Aspect", "Ar_Case", "Ar_Definite", "Ar_Gender", "Ar_Mood", "Ar_Number", "Ar_NumForm", "Ar_NumValue",
         "Ar_Person", "Ar_PronType", "Ar_Voice"],
        ["Zh_Aspect", "Zh_NumType", "Zh_Person", "Zh_Voice"],
        ["He_Case", "He_Definite", "He_Gender", "He_HebBinyan", "He_Number", "He_Person", "He_Polarity",
         "He_PronType", "He_Tense", "He_VerbForm", "He_Voice"],
        ["Hi_Aspect", "Hi_Case", "Hi_Gender", "Hi_Mood", "Hi_Number", "Hi_NumType", "Hi_Person", "Hi_PronType",
         "Hi_Tense", "Hi_VerbForm", "Hi_Voice"],
        ["Ru_Animacy", "Ru_Aspect", "Ru_Case", "Ru_Degree", "Ru_Gender", "Ru_Mood", "Ru_Number", "Ru_Person",
         "Ru_Tense", "Ru_VerbForm", "Ru_Voice"],
        ["Ta_Case", "Ta_Gender", "Ta_Mood", "Ta_Number", "Ta_NumType", "Ta_Person", "Ta_PunctType", "Ta_Tense",
         "Ta_VerbForm"]
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
    elif args.language == "English" or None:
        assert args.task_index is not None
        results = se.eval(transfer_tasks_senteval[args.task_index])
    #print(results)
    if args.language != None and args.task_index == None:
        output_path = '{}_p={}_l={}_lg={}_s={}.csv'.format(
            args.model_name,
            args.pooling,
            args.layer,
            args.language,
            params['seed'])

        pred_path = '{}_p={}_l={}_lg={}_s={}_preds.csv'.format(
            args.model_name,
            args.pooling,
            args.layer,
            args.language,
            params['seed'])

    else:
        output_path = '{}_p={}_l={}_t={}_s={}.csv'.format(
            args.model_name,
            args.pooling,
            args.layer,
            args.task_index,
            params['seed'])

        pred_path = '{}_p={}_l={}_t={}_s={}_preds.csv'.format(
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
