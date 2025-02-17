# MMRQA (Multi-Modal Retrieval Question Answering)

A proof-of-concept implementation for multi-modal document processing and semantic retrieval.

## 🛠️ Implementation Status
- [x] Core text processing pipeline
- [x] Image feature extraction
- [x] Speech-to-text conversion
- [ ] Vector database integration
- [ ] Hybrid search implementation
- [ ] API endpoints

## 📦 Installation

### System Requirements
- Python 3.9+
- FFmpeg (for audio processing)
```bash
# MacOS
brew install ffmpeg swig
```

## 🧠 Core Modules

### System Architecture
```plaintext
[User Interface]
   │
   ▼ HTTP Upload/Query
[FastAPI Backend]
   │
   ├─▶ Text Files → PyMuPDF Parsing → BERT Encoding → Vector Database
   ├─▶ Image Files → CLIP Encoding     → Vector Database
   └─▶ Voice Files → Whisper to Text → BERT Encoding → Vector Database
   │
   ▼ Query Request
[Hybrid Retrieval System] (Keyword + Vector Search)
   │
   ▼ Retrieved Context
[Generation System] (LangChain Orchestration + GPT-4)
   │
   ▼ Generated Answer
[User Interface]
```
### File Processors
```bash
processors/
├── text_processor.py    # PDF/Word parsing with BERT encoding
├── image_processor.py   # CLIP-based image feature extraction
└── audio_processor.py   # Whisper speech-to-text + BERT encoding
```

## 🔍 Test
```bash
python main.py
```
