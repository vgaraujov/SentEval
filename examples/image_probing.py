from senteval.tools.validation import SplitClassifier
from datasets import load_dataset
from PIL import Image
import numpy as np
import torch
from pixel import ViTModel, get_transforms, PangoCairoTextRenderer, get_attention_mask


def resize_model_embeddings(model):
    old_pos_embeds = model.embeddings.position_embeddings[:, : 5, :]
    model.embeddings.position_embeddings.data = old_pos_embeds.clone()
    model.config.image_size = [32, 32]
    model.image_size = [32, 32]
    model.embeddings.patch_embeddings.image_size = [32, 32]

def batcher(params, batch):
    layer = params["layer"]
    model = params["model"]
    transforms = get_transforms(
        do_resize=True,
        size=(32, 32),
        rgb=False)

    encodings = [transforms(b) for b in batch]
    attention_mask = [get_attention_mask(4, 4) for i in encodings]
    # mask = torch.stack(attention_mask)
    # batch = torch.stack(encodings)
    mask = torch.stack(attention_mask).cuda()
    batch = torch.stack(encodings).cuda()
    with torch.no_grad():
        outputs, hidden_states = model(batch, attention_mask=mask, return_dict=False)

    extended_mask = torch.cat((torch.ones(mask.shape[0], 1).cuda(), mask), -1).unsqueeze(-1)
    # extended_mask = torch.cat((torch.ones(mask.shape[0], 1), mask), -1).unsqueeze(-1)

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



dataset = load_dataset("mnist")
dataset_val = dataset['train'].train_test_split(test_size=10000)
dataset["train"] = dataset_val["train"][:50]
dataset["validation"] = dataset_val["test"][:10]
dataset["test"] = dataset["test"][:10]
print(dataset)

model = ViTModel.from_pretrained("Team-PIXEL/pixel-base")

resize_model_embeddings(model)
model.eval()

renderer_cls = PangoCairoTextRenderer

params = { "model": model, "layer": 1}

task_embed = {'train': {}, 'validation': {}, 'test': {}}
batch_size = 32

for split in dataset:
    indexes = list(range(len(dataset[split]['label'])))
    task_embed[split]['X'] = []
    # sorted_data = sorted(zip(dataset[split]['image'],
    #                          dataset[split]['label'], indexes),
    #                      key=lambda z: (len(z[0]), z[1], z[2]))
    for i in range(0,len(dataset[split]['label']), batch_size):
        batch = dataset[split]['image'][i:i+batch_size]
        embs = batcher(params, batch)
        task_embed[split]['X'].append(embs)
    task_embed[split]['X'] = np.vstack(task_embed[split]['X'])
    task_embed[split]['y'] = np.array(dataset[split]['label'])
    task_embed[split]['idx'] = np.array(indexes)

assert task_embed['train']['X'].shape[0] == task_embed['train']['y'].shape[0] == task_embed['train']['idx'].shape[0]

params_classifier = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}
config_classifier = {'nclasses': 10, 'seed': 1223,
                             'usepytorch': True,
                             'classifier': params_classifier}

clf = SplitClassifier(X={'train': task_embed['train']['X'],
                                 'valid': task_embed['validation']['X'],
                                 'test': task_embed['test']['X']},
                              y={'train': task_embed['train']['y'],
                                 'valid': task_embed['validation']['y'],
                                 'test': task_embed['test']['y']},
                              config=config_classifier)

devacc, testacc, predictions = clf.run()

print(('\nDev acc : %.1f Test acc : %.1f for %s classification\n' % (devacc, testacc, "MNIST")))




# embs = batcher(params, batch)




