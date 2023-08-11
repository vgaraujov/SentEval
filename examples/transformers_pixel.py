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

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

from PIL import Image
from pixel import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Modality,
    PangoCairoTextRenderer,
    PIXELConfig,
    ViTModel,
    PIXELTrainer,
    PIXELTrainingArguments,
    PoolingMode,
    PyGameTextRenderer,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    log_sequence_classification_predictions,
    resize_model_embeddings,
)

# SentEval prepare and batcher
def prepare(params, samples):
    pass

def batcher(params, batch):
    pooling = params["pooling"]
    layer = params["layer"]
    model = params["model"]
    processor = params.tokenizer
    format_fn = glue_strip_spaces
    transforms = get_transforms(
        do_resize=True,
        size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
    )

    batch = [[token for token in sent] for sent in batch]
    batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    encodings = [processor(text=format_fn(a)) for a in batch]
    pixel_values = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
    attention_mask = [
    get_attention_mask(e.num_text_patches, seq_length=529) for e in encodings
    ]

    with torch.no_grad():
        batch = torch.stack(pixel_values).cuda()
        mask = torch.stack(attention_mask).cuda() # bs * seq_length
        outputs, pooled_output, hidden_states, _ = model(batch, attention_mask=mask, return_dict=False)

    extended_mask = torch.cat((torch.ones(mask.shape[0], 1).cuda(), mask), -1).unsqueeze(-1)
    # extended_mask = mask.unsqueeze(-1)
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
    parser.add_argument("--model_name", default="pixel", type=str, 
                        choices=["pixel"],
                        help="the name of transformer model to evaluate on")
    parser.add_argument("--task_index", default=0, type=int,
                        help="which task to perform")
    parser.add_argument("--pooling", default="cls", type=str,
                        choices=["cls", "mean"],
                        help="which layer to evaluate on")
    parser.add_argument("--layer", default="all", type=str,
                        help="which layer to evaluate on")
    parser.add_argument("--seed", default=1111, type=int,
                        help="which seed to use")
    args = parser.parse_args()

    model_dict = {"pixel": "Team-PIXEL/pixel-base"}

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    renderer_cls = PyGameTextRenderer #if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
    processor = renderer_cls.from_pretrained(
        model_dict[args.model_name],
        rgb=False,
    )
    config = PIXELConfig.from_pretrained(model_dict[args.model_name])
    config.output_hidden_states = True
    config.output_attentions = True
    model = ViTModel.from_pretrained(model_dict[args.model_name], config=config).cuda()
    model.eval()

    # Set params for DiscoEval or SentEval
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16,
              'tokenizer': processor, "pooling": args.pooling, "layer": args.layer, "model": model, 'seed': args.seed}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}

    se = senteval.engine.SE(params, batcher, prepare)
    transfer_tasks = [
        ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC'], # stand-alone sentence classification
        ['MRPC', 'SNLI', 'SICKEntailment'], # pair-sentence clasificationc
        ['SICKRelatedness', 'STSBenchmark'], # supervised semantic similarity
        ['STS12', 'STS13', 'STS14', 'STS15', 'STS16'], # unsupervised semantic similarity
        ['Length', 'WordContent', 'Depth', 'TopConstituents',
         'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 
         'OddManOut', 'CoordinationInversion'] # probing tasks
    ]

    results = se.eval(transfer_tasks[args.task_index])
    print(results)

    output_path = '{}_p={}_l={}_t={}_s={}.pickle'.format(
        args.model_name,
        args.pooling,
        args.layer,
        args.task_index,
        params['seed'])

    # df = pd.DataFrame(results)
    # df.to_csv(output_path, index=True)

    with open(output_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
