# 📚 PaperWise - Intelligent PDF Assistant

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![Free LLM](https://img.shields.io/badge/Free%20LLM-APIs-orange.svg)](https://github.com/adawatia/PaperWise)
[![Issues](https://img.shields.io/github/issues/adawatia/PaperWise)](https://github.com/adawatia/PaperWise/issues)
[![Stars](https://img.shields.io/github/stars/adawatia/PaperWise)](https://github.com/adawatia/PaperWise/stargazers)

> Transform your PDF documents into interactive conversations with AI


**Disclaimer**: This project is built by a daily learner. Expect bugs, incomplete features, and ongoing improvements. Use at your own discretion and feel free to contribute!

PaperWise is an intelligent PDF assistant that revolutionizes how you interact with PDF documents. Ask questions, generate summaries, and extract insights from your documents using free LLM APIs.

## ✨ Why Choose PaperWise?

🆓 **Free LLM APIs** - Leverages free language model APIs for AI functionality  
🚀 **Lightning Fast** - Streamlit-powered web interface for smooth interactions  
💬 **Natural Conversations** - Chat with your PDFs like talking to an expert  
📊 **Smart Insights** - Extract key information and generate summaries  
🎨 **Beautiful Interface** - Modern, intuitive design built with Streamlit  
🌍 **Cross-Platform** - Works seamlessly on Windows, macOS, and Linux  

## 🛠️ Tech Stack

- **Python 3.8+** 🐍 - Robust backend processing
- **Streamlit** 🎨 - Modern web-based UI framework
- **Free LLM APIs** 🤖 - AI inference through free API services
- **PyMuPDF** 📄 - Advanced PDF parsing and text extraction
- **Sentence Transformers** 🔍 - Semantic search and embeddings

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed on your system
- **4GB+ RAM** recommended for optimal performance
- **Internet connection** for LLM API access

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
   streamlit run main.py
   ```

## 📖 How to Use

1. **📁 Load Your PDF**
   - Upload your PDF document through the Streamlit interface
   - Wait for processing to complete

2. **💬 Start Chatting**
   - Type questions about your document in the chat interface
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
├── main.py                 # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # MIT license file
```

## ⚙️ Configuration

### LLM API Configuration
**⚠️ Upcoming...**

PaperWise uses free LLM APIs for AI functionality. Configure your preferred API in `src/ai/llm_client.py`:

```python
# Supported free LLM APIs
SUPPORTED_APIS = [
    "huggingface",      # Hugging Face Inference API
    "together",         # Together AI (free tier)
    "replicate",        # Replicate (free tier)
]
```

### Performance Tuning

- **Response Time**: Depends on API provider and internet connection
- **Rate Limits**: Respect free tier limitations of chosen API
- **Quality**: Different APIs provide varying response quality

## 🔧 Troubleshooting

### Common Issues

**Q: Application loads slowly**  
A: Check your internet connection and API response times.

**Q: API rate limit errors**  
A: Wait for the rate limit to reset or try a different free API provider.

**Q: PDF not loading**  
A: Ensure the PDF isn't password-protected or corrupted. Scanned PDFs may need OCR.

**Q: Responses seem inaccurate**  
A: Try rephrasing your question or switch to a different LLM API for better accuracy.

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 2GB | 4GB+ |
| Storage | 1GB free | 2GB+ free |
| CPU | Dual-core | Quad-core+ |
| Internet | Broadband | High-speed |
| OS | Windows 10, macOS 10.14, Ubuntu 18.04 | Latest versions |

## 🛣️ Roadmap

### Coming Soon
- [ ] **Multi-PDF Support** - Compare and analyze multiple documents
- [ ] **OCR Integration** - Support for scanned PDFs and images
- [ ] **Export Features** - Save conversations and summaries
- [ ] **Custom Prompts** - Create reusable question templates

### Future Releases
- [ ] **Plugin System** - Extend functionality with custom modules
- [ ] **API Key Management** - Support for premium API tiers
- [ ] **Collaboration** - Share insights with team members
- [ ] **Mobile Optimization** - Better mobile web experience

## 🤝 Contributing

We love contributions! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes and add tests
5. Run tests: `pytest tests/`
6. Test the Streamlit app: `streamlit run main.py`
7. Commit: `git commit -m 'Add amazing feature'`
8. Push: `git push origin feature/amazing-feature`
9. Open a Pull Request

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

- Free LLM API providers for making AI accessible to everyone
- [Streamlit](https://streamlit.io/) for the excellent web framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) for robust PDF processing
- The open-source community for continuous inspiration

---

<div align="center">

**⭐ Star this project if you find it useful! ⭐**

[🚀 Get Started](https://github.com/adawatia/PaperWise) • [📖 Documentation](https://github.com/adawatia/PaperWise/wiki) • [🐛 Report Bug](https://github.com/adawatia/PaperWise/issues) • [💡 Request Feature](https://github.com/adawatia/PaperWise/issues)

*Built with ❤️ by [adawatia](https://github.com/adawatia) - A daily learner on the journey of improvement*

</div>
