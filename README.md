# ⚖️ Legal Clarity

> NLP-powered web app that simplifies, summarizes, and translates Indian legal documents into plain language.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-black?logo=flask)](https://flask.palletsprojects.com/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![Deploy on Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)](https://render.com/)

---

## 📖 About

**Legal Clarity** is a Flask web application that makes Indian legal documents accessible to everyone. It takes dense, jargon-heavy legal text and runs it through a three-step NLP pipeline: **legal term simplification → summarization → Hindi translation**. Built as an NLP research project, it leverages state-of-the-art Hugging Face transformer models (PEGASUS, T5, mBART-50) to break the language barrier in legal literacy.

---

## ✨ Features

- 🔍 **Legal Term Simplification** — Detects and annotates 1100+ legal terms using a custom dictionary (`LDict.py`)
- 📝 **Dual Summarization Methods**
  - **Method 1**: PEGASUS (`google/pegasus-cnn_dailymail`) — abstractive summarization
  - **Method 2**: T5 Legal (`stjiris/t5-portuguese-legal-summarization`) + T5-base paraphrasing
- 🌐 **Hindi Translation** — Translates summarized output to Hindi using `facebook/mbart-large-50-one-to-many-mmt`
- ⚡ **Auto Device Detection** — Runs seamlessly on CUDA, Apple MPS, or CPU
- 🐳 **Docker + Render Ready** — Fully containerized and deployable

---

## 🗂️ Project Structure

```
Legal_Clarity/
├── app.py              # Flask app — routes, model loading, inference logic
├── LDict.py            # Legal term dictionary + fuzzy matcher (1100+ terms)
├── templates/
│   └── index.html      # Frontend UI
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container setup
└── render.yaml         # Render deployment config
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/PATELOM925/Legal_Clarity.git
cd Legal_Clarity
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

App runs at `http://localhost:5003`

### Run with Docker

```bash
docker build -t legal-clarity .
docker run -p 5003:5003 legal-clarity
```

---

## 🧠 Models Used

| Model | Source | Purpose |
|-------|--------|---------|
| `google/pegasus-cnn_dailymail` | HuggingFace | Summarization (Method 1) |
| `stjiris/t5-portuguese-legal-summarization` | HuggingFace | Legal Summarization (Method 2) |
| `t5-base` | HuggingFace | Paraphrasing |
| `facebook/mbart-large-50-one-to-many-mmt` | HuggingFace | English → Hindi Translation |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/` | Simplify + Summarize legal text |
| `POST` | `/translate` | Translate summarized text to Hindi |

**Request body (`/`):**

```json
{ "input_text": "...", "method": "method1" }
```

**Request body (`/translate`):**

```json
{ "text": "..." }
```

---

## 📦 Key Dependencies

```
flask==3.1.0
torch==2.2.1
transformers==4.47.1
nltk==3.9.1
sentencepiece==0.2.0
gunicorn==23.0.0
numpy==1.26.0
rouge_score==0.1.2
sacrebleu==2.5.1
```

---

## 🙋 Author

**OM M PATEL**
MSc Computer Science (AI), York University
[GitHub](https://github.com/PATELOM925) · [LinkedIn](https://www.linkedin.com/in/om-m-patel/)

---

## 📄 License

This project is for academic and research purposes.
