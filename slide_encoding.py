from transformers import BeitFeatureExtractor, BeitModel
import torch
from PIL import Image

feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/dit-base")
model = BeitModel.from_pretrained("microsoft/dit-base")

def encode_slide(img_path):
    image = Image.open(img_path).convert('RGB')
    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    return outputs

if __name__ == "__main__":
    img_path = "./data/_DCBerlin18_108_Turner_REAL_TIME_AUDIO_WITH_THE_ON_J04iPJBkAKs_49008_49319.png"
    outputs = encode_slide(img_path)
    last_hidden_states = outputs.last_hidden_state
    print(list(last_hidden_states.shape))
    print(outputs.pooler_output.shape)
