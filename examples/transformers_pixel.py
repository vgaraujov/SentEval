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
import math
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

    batch = [sent.split() for sent in batch]
    # batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    encodings = [processor(text=format_fn(a)) for a in batch]
    pixel_values = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
    attention_mask = [
    get_attention_mask(e.num_text_patches, seq_length=processor.max_seq_length) for e in encodings
    ]

    with torch.no_grad():
        # batch = torch.stack(pixel_values)
        # mask = torch.stack(attention_mask)# bs * seq_length
        batch = torch.stack(pixel_values).cuda()
        mask = torch.stack(attention_mask).cuda() # bs * seq_length
        outputs, pooled_output, hidden_states, _ = model(batch, attention_mask=mask, return_dict=False)

    extended_mask = torch.cat((torch.ones(mask.shape[0], 1).cuda(), mask), -1).unsqueeze(-1)

    # extended_mask = torch.cat((torch.ones(mask.shape[0], 1), mask), -1).unsqueeze(-1)
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

def resize_model_embeddings(model: ViTModel, max_seq_length: int) -> None:
    """
    Checks whether position embeddings need to be resized. If the specified max_seq_length is longer than
    the model's number of patches per sequence, the position embeddings will be interpolated.
    If max_seq_length is shorter, the position embeddings will be truncated

    Args:
        model (`ViTModel`):
            The model for which position embeddings may be resized.
        max_seq_length (`int`):
            The maximum sequence length that determines the number of patches (excluding CLS patch) in the
            model.
    """
    patch_size = model.config.patch_size
    if isinstance(model.config.image_size, tuple) or isinstance(model.config.image_size, list):
        old_height, old_width = model.config.image_size
    else:
        old_height, old_width = (model.config.image_size, model.config.image_size)

    # ppr means patches per row (image is patchified into grid of [ppr * ppr])
    old_ppr = math.sqrt(old_height * old_width) // patch_size
    new_ppr = math.sqrt(max_seq_length)

    if old_ppr < new_ppr:
        # Interpolate position embeddings
        # logger.info(f"Interpolating position embeddings to {max_seq_length}")
        model.config.interpolate_pos_encoding = True
    elif old_ppr > new_ppr:
        # logger.info(f"Truncating position embeddings to {max_seq_length}")
        # Truncate position embeddings
        old_pos_embeds = model.embeddings.position_embeddings[:, : max_seq_length + 1, :]
        model.embeddings.position_embeddings.data = old_pos_embeds.clone()
        # Update image_size
        new_height = int(new_ppr * patch_size) if old_height == old_width else int(patch_size)
        new_width = int(new_ppr * patch_size) if old_height == old_width else int(patch_size * new_ppr ** 2)
        model.config.image_size = [new_height, new_width]
        model.image_size = [new_height, new_width]
        model.embeddings.patch_embeddings.image_size = [new_height, new_width]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name", default="pixel", type=str, 
                        choices=["pixel"],
                        help="the name of transformer model to evaluate on")
    parser.add_argument("--task_index", default=None, type=int,
                        help="which task to perform")
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
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="which max length to use")
    args = parser.parse_args()

    model_dict = {"pixel": "Team-PIXEL/pixel-base"}

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    renderer_cls = PangoCairoTextRenderer #if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
    processor = renderer_cls.from_pretrained(
        model_dict[args.model_name],
        rgb=False,
    )
    config = PIXELConfig.from_pretrained(model_dict[args.model_name])
    config.output_hidden_states = True
    config.output_attentions = True
    model = ViTModel.from_pretrained(model_dict[args.model_name], config=config).cuda()
    # model = ViTModel.from_pretrained(model_dict[args.model_name], config=config)
    model.eval()

    output_path = '{}_p={}_l={}_t={}_s={}'.format(
        args.model_name,
        args.pooling,
        args.layer,
        args.task_index,
        args.seed)

    # Set params for DiscoEval or SentEval
    # params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16,
    #           'tokenizer': processor, 'pooling': args.pooling, 'layer': args.layer, 'model': model,
    #           'seed': args.seed, 'save_emb': output_path}

    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16,
              'tokenizer': processor, 'pooling': args.pooling, 'layer': args.layer, 'model': model,
              'seed': args.seed, 'save_emb': None}
    
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}

    se = senteval.engine.SE(params, batcher, prepare)
    transfer_tasks_senteval = [
        ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC'],  # stand-alone sentence classification
        ['MRPC', 'SNLI', 'SICKEntailment'],  # pair-sentence clasificationc
        ['SICKRelatedness', 'STSBenchmark'],  # supervised semantic similarity
        ['STS12', 'STS13', 'STS14', 'STS15', 'STS16'],  # unsupervised semantic similarity
        ['Length', 'WordContent', 'Depth', 'TopConstituents',
         'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
         'OddManOut', 'CoordinationInversion'],  # probing tasks
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
    # print(results)
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

    # deprecation: move to pickle format for ease
    # df = pd.DataFrame(results)
    # df.to_csv(output_path+'.csv', index=True)


