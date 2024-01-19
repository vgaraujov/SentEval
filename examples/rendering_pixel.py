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
sents = ["שוטר וחייל אחזו בדלתות האמבולנס , כשפניהם אל המתפרעים , כדי לסוכך על הפצוע הערבי"]
out = [processor(text=sent) for sent in tqdm(sents)]
# transforms = get_transforms(
#         do_resize=True,
#         size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
#     )
# pixel_values = [transforms(Image.fromarray(e.pixel_values)) for e in out]

#save out to a pickle file
new_image = Image.fromarray(out[0].pixel_values)
# new_image = transforms(new_image)
new_image.save("test-pango-hebrew.png")
