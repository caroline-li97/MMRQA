# test_env.py
import fitz  # PyMuPDF
from transformers import BertTokenizer
import whisper
import torch

print("--> PDF test pass" if hasattr(fitz, 'open') else "× PDF dependency error")
print("--> BERT test pass" if BertTokenizer.from_pretrained('bert-base-uncased') else "× BERT error")
print("--> Whisper test pas" if whisper.load_model('base') else "× voice error")
print("--> Torch CUDA test pass" if torch.cuda.is_available() else "! CPU processing")