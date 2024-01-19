from PIL import Image
from tqdm import tqdm
import pickle
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
renderer_cls = PangoCairoTextRenderer
processor = renderer_cls.from_pretrained(
        "Team-PIXEL/pixel-base",
        rgb=False,
    )

with open("../data/probing/Hindi/Aspect.txt", "r") as f:
    text = f.read()

sents = []
lines = text.split('\n')
for line in lines:
    if line != '':
        line = line.split('\t')
        sents.append(line[2])

encodings = {}
for sent in tqdm(sents):
    out = processor(text=sent)
    Image.fromarray(out.pixel_values)
    encodings[sent] = out

##See the rendered image here########

# out = [processor(text=sent) for sent in tqdm(sents)]
#
# #save out to a pickle file
# new_image = Image.fromarray(out[0].pixel_values)
# # new_image = transforms(new_image)
# new_image.save("test-pango-hebrew.png")
