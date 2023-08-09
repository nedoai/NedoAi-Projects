from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

#Your text list for meme
with open(r"Meme-gen/memes.txt", "r", encoding="utf-8") as f:
    memes_text = [line.strip() for line in f]
    print("memes.txt loaded")

def meme_text_gen(image_path):
    #Load pretrained model CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    image = Image.open(image_path)

    text_inputs = processor(memes_text, return_tensors="pt", padding=True, truncation=True)
    image_inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        image_features = model.get_image_features(**image_inputs)

    similarity_scores = (100.0 * image_features @ text_features.T).softmax(dim=1)[0]

    best_idx = similarity_scores.argmax()
    best_meme_text = memes_text[best_idx]
    best_similarity = similarity_scores[best_idx].item()

    print(f"Generated Meme: {best_meme_text}")
    print(f"Similarity: {best_similarity:.2f}%")

    return best_meme_text, best_similarity

print(meme_text_gen(r"Meme-gen/photo.jpg"))