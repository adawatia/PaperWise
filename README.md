# 📚 PaperWise - Intelligent PDF Assistant  

PaperWise is an AI-powered PDF assistant that enables smart interaction with documents. It allows users to extract insights, ask questions, and summarize content efficiently while running offline with **Ollama**.

---

## 🚀 Features  

✅ **Smart Q&A:** Extract relevant answers from PDFs  
✅ **Summarization:** Get concise summaries of lengthy documents  
✅ **Local & Offline:** Powered by Ollama for privacy and security  
✅ **User-Friendly UI:** Built with PySide6 for a smooth experience  
✅ **Efficient Processing:** Uses PyMuPDF for fast text extraction  

---

## 🛠️ Tech Stack  

- **Python** 🐍 - Core programming language  
- **PySide6** 🎨 - GUI framework for a responsive UI  
- **Ollama** 🤖 - Runs large language models locally  
- **PyMuPDF** 📄 - PDF parsing and text extraction  
- **Requests** 🔗 - API communication and network operations  

---

## 📂 Project Structure  

```
paperwise/
│
├── main.py                     # Application entry point
├── requirements.txt            # Dependencies list
│
├── paperwise/                  # Main package
│   ├── core/                   # Core functionality 
│   │   ├── pdf_processor.py    # PDF loading and text extraction
│   │   ├── query_processor.py  # Query handling and context retrieval
│   │   └── ollama_interface.py # LLM interaction via Ollama
│   │
│   ├── ui/                     # User interface components
│   │   ├── main_window.py      # Main application window
│   │   ├── pdf_viewer.py       # PDF viewing component
│   │   ├── query_panel.py      # Query input and response display
│   │   └── document_list.py    # Document management sidebar
│   │
│   └── utils/                  # Utility functions
│       ├── text_chunking.py    # Text chunking algorithms
│       └── config.py           # Application configuration
│
├── resources/                  # UI resources (icons, styles)
├── tests/                      # Unit and integration tests
└── document_storage/           # Default storage location for processed documents
```

---

## 🔧 Installation  

1️⃣ **Clone the repository:**  
```bash
git clone https://github.com/adawatia/PaperWise.git
cd PaperWise
```

2️⃣ **Set up a virtual environment:**  
```bash
uv venv --python 3.13
```

3️⃣ **Install dependencies:**  
```bash
uv pip install -r requirements.txt
```

4️⃣ **Run the application:**  
```bash
uv run main.py
```

---

## 🚀 Upcoming Features  

🔹 Multi-document support  
🔹 Advanced AI-based summarization  
🔹 OCR for scanned PDFs  
🔹 Enhanced UI/UX improvements  

---

## 🤝 Contributing  

We welcome contributions! Feel free to submit pull requests, report issues, or suggest new features.

---

## 📜 License  

This project is licensed under the **MIT License**.  

---

⭐ If you find **PaperWise** useful, give it a **star** on GitHub! 🚀  

---
