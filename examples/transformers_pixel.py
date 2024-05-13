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
    PangoCairoBigramsRenderer,
    PIXELConfig,
    ViTModel,
    PIXELTrainer,
    PIXELTrainingArguments,
    PoolingMode,
    PyGameTextRenderer,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    log_sequence_classification_predictions
)

# SentEval prepare and batcher
def prepare(params, samples):
    pass

def batcher(params, batch):
    pooling = params["pooling"]
    layer = params["layer"]
    model = params["model"]
    model_name = params["model_name"]
    processor = params.tokenizer
    format_fn = glue_strip_spaces
    if "vit" in model.config._name_or_path:
        transforms = get_transforms(
            do_resize=False,
            do_squarify=True
        )
    else:
        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )

    # batch = [[token for token in sent] for sent in batch]
    # batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    if model_name == "pixel-words" or model_name == "pixel_r" or model_name == 'pixel-bigrams':
        encodings = [processor(text=a.split()) for a in batch]
    else:
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
                        choices=["pixel", "mpixel", "vit-mae", "pixel-words", "pixel-r", "pixel-bigrams", "pixel-bigrams-r", "pixel-words-r"],
                        help="the name of transformer model to evaluate on")
    parser.add_argument("--task_index", default=None, type=int,
                        help="which task to perform")
    parser.add_argument("--language", default=None, type=str,
                        choices=["Arabic", "Chinese", "Hebrew", "Hindi", "Russian", "Tamil", "Korean", "Japanese",
                                 "English", "English_UD", "Coptic", "Sanskrit",
                                 "Xru", "Xde", "Xes", "Xfi", "Xfr", "Xtr", "Visual"])
    parser.add_argument("--pooling", default="cls", type=str,
                        choices=["cls", "mean"],
                        help="which layer to evaluate on")
    parser.add_argument("--layer", default="all", type=str,
                        help="which layer to evaluate on")
    parser.add_argument("--seed", default=1111, type=int,
                        help="which seed to use")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="which max length to use")
    parser.add_argument("--auth", default=None, type=str,
                        help="hf authentication token")
    args = parser.parse_args()

    model_dict = {"pixel": "Team-PIXEL/pixel-base", "mpixel": "Team-PIXEL/mpixel-base2", "vit-mae": "facebook/vit-mae-base",
                  "pixel-r": "Team-PIXEL/pixel-base", "pixel-words": "Team-PIXEL/pixel-small-words", "pixel-bigrams": "Team-PIXEL/pixel-base-bigrams",
                  "pixel-words-r": "Team-PIXEL/pixel-small-words", "pixel-bigrams-r": "Team-PIXEL/pixel-base-bigrams"}
    access_token = args.auth

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    if args.model_name == "pixel-bigrams":
        renderer_cls = PangoCairoBigramsRenderer
    else:
        renderer_cls = PangoCairoTextRenderer
    if args.model_name == "vit-mae":
        processor = renderer_cls.from_pretrained(
            model_dict["pixel"],
            rgb=False,
            max_seq_length=196,
            fallback_fonts_dir="fallback_fonts",
            use_auth_token=access_token
        )
    elif args.model_name == "pixel-bigrams":
        processor = renderer_cls.from_pretrained(
            "test_text_renderer_config.json",
            rgb=False,
            max_seq_length=args.max_seq_length,
            fallback_fonts_dir="fallback_fonts"
        )
    else:
        processor = renderer_cls.from_pretrained(
            model_dict[args.model_name],
            rgb=False,
            max_seq_length=args.max_seq_length,
            fallback_fonts_dir="fallback_fonts",
            use_auth_token=access_token
        )

    config = PIXELConfig.from_pretrained(model_dict[args.model_name], use_auth_token=access_token)
    config.output_hidden_states = True
    config.output_attentions = True
    model = ViTModel.from_pretrained(model_dict[args.model_name], config=config, use_auth_token=access_token).cuda()
    # model = ViTModel.from_pretrained(model_dict[args.model_name], config=config)
    if "pixel" in args.model_name:
        resize_model_embeddings(model, args.max_seq_length)
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
              'model_name': args.model_name,
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
         'Xtr_OddManOut', 'Xtr_CoordinationInversion'],
        ["Vis_MaxCharacter"]

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
    elif args.language == "Visual":
        results = se.eval(transfer_tasks[15])
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


