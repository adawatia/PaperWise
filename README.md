# 📚 PaperWise - Intelligent PDF Assistant

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flet](https://img.shields.io/badge/Flet-Latest-purple.svg)](https://flet.dev/)
[![GPT4All](https://img.shields.io/badge/GPT4All-Powered-orange.svg)](https://gpt4all.io/)
[![Issues](https://img.shields.io/github/issues/adawatia/PaperWise)](https://github.com/adawatia/PaperWise/issues)
[![Stars](https://img.shields.io/github/stars/adawatia/PaperWise)](https://github.com/adawatia/PaperWise/stargazers)

> Transform your PDF documents into interactive conversations with AI

PaperWise is a privacy-first, offline AI assistant that revolutionizes how you interact with PDF documents. Ask questions, generate summaries, and extract insights from your documents - all while keeping your data completely private on your local machine.

## ✨ Why Choose PaperWise?

🔒 **100% Privacy** - Your documents never leave your computer  
🚀 **Lightning Fast** - Local processing with no internet dependency  
💬 **Natural Conversations** - Chat with your PDFs like talking to an expert  
📊 **Smart Insights** - Extract key information and generate summaries  
🎨 **Beautiful Interface** - Modern, intuitive design built with Flet  
🌍 **Cross-Platform** - Works seamlessly on Windows, macOS, and Linux  

## 🛠️ Tech Stack

- **Python 3.8+** 🐍 - Robust backend processing
- **Flet** 🎨 - Modern GUI framework (Flutter-powered)
- **GPT4All** 🤖 - Local AI inference engine
- **PyMuPDF** 📄 - Advanced PDF parsing and text extraction
- **Sentence Transformers** 🔍 - Semantic search and embeddings

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed on your system
- **8GB+ RAM** recommended for optimal performance
- **2GB+ free disk space** for AI models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/adawatia/PaperWise.git
   cd PaperWise
   ```

2. **Set up virtual environment** (recommended)
   ```bash
   # Using uv (recommended)
   uv venv --python 3.13
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Or using standard venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Using uv
   uv pip install -r requirements.txt
   
   # Or using pip
   pip install -r requirements.txt
   ```

4. **Launch PaperWise**
   ```bash
   python main.py
   ```

## 📖 How to Use

1. **📁 Load Your PDF**
   - Click "Open PDF" and select your document
   - Wait for processing to complete

2. **💬 Start Chatting**
   - Type questions about your document
   - Ask for summaries of specific sections
   - Request key insights and analysis

3. **🎯 Example Queries**
   - "What are the main conclusions of this paper?"
   - "Summarize the methodology section"
   - "What are the key findings related to [topic]?"
   - "List the references mentioned about [subject]"

## 📁 Project Structure

```
paperwise/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # MIT license file
├── src/                   # Source code
│   ├── ui/               # User interface components
│   │   ├── __init__.py
│   │   └── main_window.py
│   ├── pdf/              # PDF processing logic
│   │   ├── __init__.py
│   │   ├── parser.py     # PDF text extraction
│   │   └── processor.py  # Text preprocessing
│   ├── ai/               # AI integration
│   │   ├── __init__.py
│   │   ├── gpt4all_client.py  # GPT4All interface
│   │   └── embeddings.py      # Text embeddings
│   └── utils/            # Utility functions
│       ├── __init__.py
│       └── helpers.py
├── assets/               # Static resources
│   ├── icons/
│   └── fonts/
├── models/              # Downloaded AI models (auto-created)
├── tests/               # Unit tests
│   ├── __init__.py
│   ├── test_pdf.py
│   └── test_ai.py
└── docs/                # Additional documentation
    └── API.md
```

## ⚙️ Configuration

### AI Model Selection

PaperWise automatically downloads and uses the best available model for your system. You can customize the model in `src/ai/gpt4all_client.py`:

```python
# Recommended models (in order of preference)
MODELS = [
    "mistral-7b-openorca.Q4_0.gguf",    # Balanced performance
    "orca-mini-3b.gguf",                # Faster, less accurate
    "wizardlm-13b-v1.2.Q4_0.gguf"      # Higher accuracy, slower
]
```

### Performance Tuning

- **RAM Usage**: Models range from 2GB (3B parameters) to 8GB+ (13B+ parameters)
- **Processing Speed**: Smaller models respond faster but may be less accurate
- **Quality**: Larger models provide better understanding and responses

## 🔧 Troubleshooting

### Common Issues

**Q: Application starts slowly**  
A: First launch downloads AI models (~2-4GB). Subsequent starts are much faster.

**Q: Out of memory errors**  
A: Try a smaller model or close other applications to free up RAM.

**Q: PDF not loading**  
A: Ensure the PDF isn't password-protected or corrupted. Scanned PDFs may need OCR.

**Q: Responses seem inaccurate**  
A: Try rephrasing your question or use a larger model for better accuracy.

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| Storage | 5GB free | 10GB+ free |
| CPU | Dual-core | Quad-core+ |
| OS | Windows 10, macOS 10.14, Ubuntu 18.04 | Latest versions |

## 🛣️ Roadmap

### Coming Soon
- [ ] **Multi-PDF Support** - Compare and analyze multiple documents
- [ ] **OCR Integration** - Support for scanned PDFs and images
- [ ] **Export Features** - Save conversations and summaries
- [ ] **Custom Prompts** - Create reusable question templates

### Future Releases
- [ ] **Plugin System** - Extend functionality with custom modules
- [ ] **Cloud Sync** - Optional encrypted cloud backup
- [ ] **Collaboration** - Share insights with team members
- [ ] **Mobile App** - iOS and Android companion apps

## 🤝 Contributing

We love contributions! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes and add tests
5. Run tests: `pytest tests/`
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions
- Write tests for new features

## 🐛 Bug Reports & Feature Requests

Help us improve PaperWise! When reporting issues:

**For Bugs:**
- [ ] Clear description of the problem
- [ ] Steps to reproduce
- [ ] Expected vs actual behavior
- [ ] System information (OS, Python version)
- [ ] Error messages or logs

**For Features:**
- [ ] Describe the use case
- [ ] Explain the expected behavior
- [ ] Consider implementation complexity

[Create an Issue](https://github.com/adawatia/PaperWise/issues/new/choose)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GPT4All](https://gpt4all.io/) for providing excellent local AI capabilities
- [Flet](https://flet.dev/) for the beautiful cross-platform UI framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) for robust PDF processing
- The open-source community for continuous inspiration

---

<div align="center">

**⭐ Star this project if you find it useful! ⭐**

[🚀 Get Started](https://github.com/adawatia/PaperWise) • [📖 Documentation](https://github.com/adawatia/PaperWise/wiki) • [🐛 Report Bug](https://github.com/adawatia/PaperWise/issues) • [💡 Request Feature](https://github.com/adawatia/PaperWise/issues)

*Built with ❤️ by [adawatia](https://github.com/adawatia)*

</div>
