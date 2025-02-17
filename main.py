# main.py
from processors.text_processor import TextProcessor
from processors.image_processor import ImageProcessor
from processors.audio_processor import AudioProcessor
import os

os.makedirs("uploads", exist_ok=True)

def test_text_processing():
    print("\ntesting text processor...")
    processor = TextProcessor()
    
    sample_pdf = "uploads/test.pdf"
    text = processor.read_file(sample_pdf)
    print(f"read {len(text)} characters")
    
    chunks = processor.chuck_text(text)
    vector = processor.encode(chunks[0])
    print(f"generate vec: {vector.shape}")

def test_image_processing():
    print("\ntesting image processor...")
    processor = ImageProcessor()
    
    sample_image = "uploads/test.jpg"
    vector = processor.encode(sample_image)
    print(f"CLIP vec: {vector.shape}")

if __name__ == "__main__":
    test_text_processing()
    test_image_processing()