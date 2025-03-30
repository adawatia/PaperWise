# 📚 PaperWise - Intelligent PDF Assistant

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flet](https://img.shields.io/badge/Flet-Latest-purple.svg)](https://flet.dev/)
[![Ollama](https://img.shields.io/badge/Ollama-Powered-orange.svg)](https://ollama.com/)
[![Issues](https://img.shields.io/github/issues/adawatia/PaperWise)](https://github.com/adawatia/PaperWise/issues)
[![Stars](https://img.shields.io/github/stars/adawatia/PaperWise)](https://github.com/adawatia/PaperWise/stargazers)

PaperWise is an AI-powered PDF assistant that enables smart interaction with documents. It allows users to extract insights, ask questions, and summarize content efficiently while running completely offline with **Ollama**.

## ✨ Key Features

- **Natural Language Querying**: Ask questions about your PDF documents in plain English
- **Smart Summarization**: Generate concise summaries of lengthy documents
- **Offline Operation**: All processing happens locally - no data leaves your machine
- **User-Friendly Interface**: Intuitive UI built with Flet for a native app experience
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 🛠️ Tech Stack

- **Python** 🐍 - Core programming language
- **Flet** 🎨 - GUI framework *(Flutter Wrapper)* for a responsive UI
- **Ollama** 🤖 - Runs large language models locally
- **PyMuPDF** 📄 - PDF parsing and text extraction
- **Requests** 🔗 - API communication and network operations

## 📋 Requirements

- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed and running
- At least 8GB RAM recommended for optimal performance

## 📂 Project Structure

```
paperwise/
│
├── main.py                     # Application entry point
├── src/                        # Source code directory
│   ├── ui/                     # UI components
│   ├── pdf/                    # PDF processing logic
│   └── ollama/                 # Ollama integration
├── assets/                     # Static assets and resources
└── tests/                      # Unit and integration tests
```

## 🔧 Installation

### 1️⃣ Prerequisites

- Ensure [Ollama](https://ollama.com/download) is installed and running
- Install Python 3.8+ and uv package manager

### 2️⃣ Clone the repository:

```bash
git clone https://github.com/adawatia/PaperWise.git
cd PaperWise
```

### 3️⃣ Set up a virtual environment:

```bash
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4️⃣ Install dependencies:

```bash
uv pip install -r requirements.txt
```

### 5️⃣ Run the application:

```bash
uv run main.py
```

## 🧩 Usage

1. Launch PaperWise using the installation steps above
2. Click "Open PDF" to load your document
3. Once loaded, you can:
   - Ask questions about the document's content
   - Generate summaries of sections or the entire document
   - Extract key information automatically

## 📊 Performance Notes

PaperWise performance depends on:
- The Ollama model you choose (larger models = better results but slower processing)
- Your system specifications (CPU, RAM, and disk speed)
- PDF complexity and length

## 🚀 Upcoming Features

- 📄 Multi-document support with cross-referencing
- 🧠 Advanced AI-based summarization with customizable length and focus
- 👁️ OCR for scanned PDFs and image-based documents
- 🎨 Enhanced UI/UX improvements with dark mode support
- 📊 Data visualization for document insights
- 🔍 Advanced search capabilities

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows our style guidelines and includes appropriate tests.

## 🐛 Issue Reporting

Found a bug or have a feature request? Please [create an issue](https://github.com/adawatia/PaperWise/issues/new) with:
- A clear description of the problem/request
- Steps to reproduce (for bugs)
- Your environment details (OS, Python version, etc.)

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

⭐ If you find **PaperWise** useful, give it a **star** on GitHub! 🚀

[GitHub Repository](https://github.com/adawatia/PaperWise) | [Report Issues](https://github.com/adawatia/PaperWise/issues)