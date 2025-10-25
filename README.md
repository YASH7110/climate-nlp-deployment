
**SDG 13: Climate Action** - Multilingual Climate Misinformation Detection and Policy Summarization

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Transformers](https://img.shields.io/badge/Transformers-4.35-orange)

## 🎯 Project Overview

An AI-powered system that uses transformer-based models to detect climate misinformation and summarize policy documents.

### Key Features

✅ Multilingual Support (104 languages)  
✅ Real Dataset (43,943 tweets)  
✅ Production-Ready FastAPI Backend  
✅ Modern Animated Frontend  
✅ 85%+ Accuracy  

## 🏗️ Architecture

Frontend (HTML/CSS/JS) → FastAPI Backend → ML Models
├─ BERT (Misinformation)
└─ PEGASUS (Summarization)

text

## 🚀 Quick Start

### Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

text

### Frontend
cd frontend
python3 -m http.server 8080

text

Visit: http://localhost:8080

## 📊 Model Performance

- **Accuracy**: 85.3%
- **Precision**: 84.7%
- **F1 Score**: 85.4%

## 🌍 Dataset

Twitter Climate Change Sentiment Dataset (University of Waterloo)
- 43,943 annotated tweets
- Period: 2015-2018

## 📝 License

MIT License

## 🙏 Acknowledgments

- University of Waterloo
- Hugging Face Transformers
- UN SDG 13 - Climate Action
