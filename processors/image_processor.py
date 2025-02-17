import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class ImageProcessor:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    def encode(self, image_path):
        """Generate image feature vector
        Workflow:
        1. Open image file
        2. CLIP preprocessing
        3. Inference to get features
        """
        try:
            image = Image.open(image_path)  # Load image
            inputs = self.processor(images=image, return_tensors="pt")  # Pixel normalization
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            return features.numpy()[0]
        except Exception as e:
            print(f"image processing failed: {str(e)}")
            return None