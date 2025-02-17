import os
from transformers import BertTokenizer, BertModel
import fitz #PyMuPDF
from docx import Document
import torch


class TextProcessor:
    
    def __init__(self):
        """Initialize BERT model"""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Text tokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased')          # BERT model instance
    
    def read_file(self, file_path):
        # File format validation
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file {file_path} doesn't exist.")
        if file_path.endswith('.pdf'):
            return self._read_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._read_docx(file_path)
        else:
            raise ValueError("Unsupported file format")  # Exception handling
    
    def _read_pdf(self, file_path):
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    
    def _read_docx(self, file_path):
        text = ""
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += "\n".join(para.text)
        return text
    
    def chuck_text(self, text, chunk_size = 256):
        chuck = []
        for i in range(0, len(text), chunk_size):
            chuck.append(text[i:i+chunk_size])
        return chuck
    
    def encode(self, text_chunk):
        inputs = self.tokenizer(
            text_chunk,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = 256
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim = 1).numpy()[0]
        
    