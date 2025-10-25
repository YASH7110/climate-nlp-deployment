from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
import numpy as np
import json
import logging
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="🌍 Climate Change NLP API",
    description="SDG 13: Climate Action - Misinformation Detection & Policy Summarization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️  Using device: {self.device}")
        
        self.misinfo_model = None
        self.misinfo_tokenizer = None
        self.summary_model = None
        self.summary_tokenizer = None
        self.metadata = None
        
        self.load_models()
    
    def load_models(self):
        """Load all models from disk"""
        try:
            # Load misinformation detector
            logger.info("📥 Loading misinformation detector...")
            self.misinfo_tokenizer = AutoTokenizer.from_pretrained(
                str(config.MISINFO_MODEL_PATH)
            )
            self.misinfo_model = AutoModelForSequenceClassification.from_pretrained(
                str(config.MISINFO_MODEL_PATH)
            ).to(self.device)
            self.misinfo_model.eval()
            logger.info("✅ Misinformation detector loaded")
            
            # Load policy summarizer (with slow tokenizer fix)
            logger.info("📥 Loading policy summarizer...")
            self.summary_tokenizer = AutoTokenizer.from_pretrained(
                str(config.SUMMARY_MODEL_PATH),
                use_fast=False  # Fix for tokenizer compatibility
            )
            self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(config.SUMMARY_MODEL_PATH)
            ).to(self.device)
            self.summary_model.eval()
            logger.info("✅ Policy summarizer loaded")
            
            # Load metadata (optional - handle errors gracefully)
            try:
                if config.METADATA_PATH.exists():
                    with open(config.METADATA_PATH, 'r') as f:
                        content = f.read().strip()
                        if content:  # Check if file is not empty
                            self.metadata = json.loads(content)
                            logger.info("✅ Metadata loaded")
                        else:
                            logger.info("⚠️  Metadata file is empty (optional)")
                            self.metadata = None
                else:
                    logger.info("⚠️  Metadata file not found (optional)")
                    self.metadata = None
            except Exception as e:
                logger.warning(f"⚠️  Could not load metadata: {e} (optional, continuing...)")
                self.metadata = None
            
            logger.info("🎉 All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def detect_misinformation(self, text: str) -> Dict:
        """Detect misinformation in climate-related text"""
        try:
            inputs = self.misinfo_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.misinfo_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item()
            
            is_misinformation = prediction == 1
            label = "misinformation" if is_misinformation else "credible"
            
            return {
                "text": text,
                "prediction": "⚠️ Potential Misinformation" if is_misinformation else "✅ Credible Information",
                "label": label,
                "confidence": float(confidence),
                "probabilities": {
                    "credible": float(probabilities[0][0]),
                    "misinformation": float(probabilities[0][1])
                },
                "explanation": self._get_explanation(is_misinformation, confidence)
            }
        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise
    
    def summarize_policy(self, document: str, max_length: int, min_length: int) -> Dict:
        """Generate abstractive summary of policy document"""
        try:
            inputs = self.summary_tokenizer(
                document,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.summary_model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            summary_text = self.summary_tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            
            original_length = len(document.split())
            summary_length = len(summary_text.split())
            compression_ratio = ((original_length - summary_length) / original_length) * 100
            
            return {
                "original_text": document,
                "summary": summary_text,
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": float(compression_ratio)
            }
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            raise
    
    def get_attention_weights(self, text: str) -> Dict:
        """Extract attention weights for visualization"""
        try:
            inputs = self.misinfo_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.MAX_LENGTH
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.misinfo_model(**inputs, output_attentions=True)
            
            attention = outputs.attentions[-1]
            attention_avg = attention[0].mean(dim=0).cpu().numpy()
            
            tokens = self.misinfo_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            attention_scores = attention_avg.sum(axis=0)
            token_importance = [
                {"token": token, "score": float(score)}
                for token, score in zip(tokens, attention_scores)
                if token not in ["[CLS]", "[SEP]", "[PAD]"]
            ]
            token_importance.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "text": text,
                "tokens": tokens,
                "key_terms": token_importance[:15],
                "attention_matrix": attention_avg.tolist()
            }
        except Exception as e:
            logger.error(f"Attention error: {e}")
            raise
    
    def _get_explanation(self, is_misinformation: bool, confidence: float) -> str:
        """Generate human-readable explanation"""
        if is_misinformation:
            if confidence > 0.9:
                return "This content shows strong patterns of climate misinformation. Please verify with authoritative scientific sources like IPCC reports."
            elif confidence > 0.7:
                return "This content likely contains climate misinformation. Cross-check with peer-reviewed scientific literature."
            else:
                return "This content may contain questionable climate claims. Further fact-checking recommended."
        else:
            if confidence > 0.9:
                return "This content strongly aligns with established climate science consensus."
            elif confidence > 0.7:
                return "This content appears credible and consistent with climate research findings."
            else:
                return "This content generally aligns with climate science, but additional verification recommended."

# Initialize model manager
model_manager = ModelManager()

# ============================================================================
# Request/Response Models
# ============================================================================

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    language: Optional[str] = Field("en", description="Language code")

class MisinfoResponse(BaseModel):
    text: str
    prediction: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    explanation: str

class SummaryInput(BaseModel):
    document: str = Field(..., min_length=100, description="Policy document to summarize")
    max_length: Optional[int] = Field(150, ge=50, le=300, description="Maximum summary length")
    min_length: Optional[int] = Field(40, ge=20, le=100, description="Minimum summary length")

class SummaryResponse(BaseModel):
    original_text: str
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float

class AttentionResponse(BaseModel):
    text: str
    tokens: List[str]
    key_terms: List[Dict[str, float]]
    attention_matrix: List[List[float]]

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., max_items=10, description="List of texts to analyze")

class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: Dict[str, bool]
    metadata: Optional[Dict]

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🌍 Climate Change NLP API - SDG 13: Climate Action",
        "version": "1.0.0",
        "description": "Multilingual climate misinformation detection and policy document summarization",
        "endpoints": {
            "misinformation_detection": "/api/detect-misinformation",
            "policy_summarization": "/api/summarize-policy",
            "attention_visualization": "/api/attention-visualization",
            "batch_detection": "/api/batch-detect",
            "health_check": "/api/health",
            "documentation": "/docs"
        }
    }

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "device": model_manager.device,
        "models_loaded": {
            "misinformation_detector": model_manager.misinfo_model is not None,
            "policy_summarizer": model_manager.summary_model is not None
        },
        "metadata": model_manager.metadata
    }

@app.post("/api/detect-misinformation", response_model=MisinfoResponse, tags=["Detection"])
async def detect_misinformation(input_data: TextInput):
    """
    Detect climate change misinformation in multilingual text.
    
    - **text**: Text to analyze (max 5000 characters)
    - **language**: Language code (optional, auto-detected)
    
    Returns prediction with confidence score and explanation.
    """
    try:
        result = model_manager.detect_misinformation(input_data.text)
        return result
    except Exception as e:
        logger.error(f"Detection endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/api/summarize-policy", response_model=SummaryResponse, tags=["Summarization"])
async def summarize_policy(input_data: SummaryInput):
    """
    Generate abstractive summary of climate policy documents.
    
    - **document**: Policy document text (min 100 characters)
    - **max_length**: Maximum summary length in tokens (50-300)
    - **min_length**: Minimum summary length in tokens (20-100)
    
    Returns concise summary with compression metrics.
    """
    try:
        result = model_manager.summarize_policy(
            input_data.document,
            input_data.max_length,
            input_data.min_length
        )
        return result
    except Exception as e:
        logger.error(f"Summarization endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

@app.post("/api/attention-visualization", response_model=AttentionResponse, tags=["Visualization"])
async def get_attention_visualization(input_data: TextInput):
    """
    Extract attention weights and key terms from text.
    
    - **text**: Text to analyze
    
    Returns tokens, attention weights, and ranked key terms.
    """
    try:
        result = model_manager.get_attention_weights(input_data.text)
        return result
    except Exception as e:
        logger.error(f"Attention endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Attention error: {str(e)}")

@app.post("/api/batch-detect", tags=["Detection"])
async def batch_detect_misinformation(input_data: BatchTextInput):
    """
    Batch misinformation detection for multiple texts.
    
    - **texts**: List of texts to analyze (max 10)
    
    Returns predictions for all texts with summary statistics.
    """
    try:
        results = []
        for text in input_data.texts:
            result = model_manager.detect_misinformation(text)
            results.append(result)
        
        return {
            "total_processed": len(results),
            "results": results,
            "summary": {
                "credible_count": sum(1 for r in results if r['label'] == 'credible'),
                "misinformation_count": sum(1 for r in results if r['label'] == 'misinformation')
            }
        }
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.get("/api/stats", tags=["System"])
async def get_stats():
    """Get model statistics and metadata"""
    if model_manager.metadata:
        return {
            "project": model_manager.metadata.get("project"),
            "sdg": model_manager.metadata.get("sdg"),
            "dataset_info": model_manager.metadata.get("dataset"),
            "model_performance": model_manager.metadata.get("models", {}).get("misinformation_detector", {}).get("performance"),
            "deployment_date": model_manager.metadata.get("deployment_date")
        }
    return {
        "message": "No metadata available",
        "note": "Models are functional but training statistics were not saved"
    }

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=config.HOST, 
        port=config.PORT, 
        reload=config.RELOAD,
        log_level="info"
    )
