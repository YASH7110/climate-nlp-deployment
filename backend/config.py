from pathlib import Path

class Config:
    """Configuration for Climate NLP API"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    
    # Model paths
    MISINFO_MODEL_PATH = MODELS_DIR / "misinformation_detector"
    SUMMARY_MODEL_PATH = MODELS_DIR / "policy_summarizer"
    ATTENTION_WEIGHTS_PATH = MODELS_DIR / "attention_weights.json"
    METADATA_PATH = MODELS_DIR / "deployment_metadata.json"
    
    # Model settings
    MAX_LENGTH = 128
    SUMMARY_MAX_LENGTH = 150
    SUMMARY_MIN_LENGTH = 40
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    RELOAD = True
    
    # CORS
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5500",  # Live Server
        "*"  # Allow all origins (for development)
    ]

config = Config()
