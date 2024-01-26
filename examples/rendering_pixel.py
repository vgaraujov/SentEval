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

# with open("../data/probing/Hindi/Aspect.txt", "r") as f:
#     text = f.read()
#
# sents = []
# lines = text.split('\n')
# for line in lines:
#     if line != '':
#         line = line.split('\t')
#         sents.append(line[2])
#
# encodings = {}
# for sent in tqdm(sents):
#     out = processor(text=sent)
#     Image.fromarray(out.pixel_values)
#     encodings[sent] = out

##See the rendered image here########
format_fn = glue_strip_spaces
transforms = get_transforms(
        do_resize=True,
        size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
    )
sents = ["My name is Anthony Gonsalves."]
batch = [[token for token in sent] for sent in sents]
batch = [" ".join(sent) if sent != [] else "." for sent in batch]
print(batch)
encodings = [processor(text=format_fn(a)) for a in batch]
new_image = Image.fromarray(encodings[0].pixel_values)
pixel_values = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
print(pixel_values)
#save out to a pickle file
# new_image = Image.fromarray(pixel_values)
new_image.save("test-batch-english.png")
# new_image = transforms(new_image)
# new_image.save("test-pango-english.png")
