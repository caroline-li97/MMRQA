# MMRQA

## System Architecture
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
