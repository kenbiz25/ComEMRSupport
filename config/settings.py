# config/settings.py
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

def _to_bool(val, default=True):
    """Convert environment variable to boolean"""
    if val is None:
        return default
    return str(val).strip().lower() in ("true", "1", "yes", "y")

def _to_float(val, default=0.55):
    """Convert environment variable to float with fallback"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def _to_int(val, default=3):
    """Convert environment variable to int with fallback"""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

class ConfidenceThresholds:
    """
    Confidence score thresholds for response strategies
    
    Strategy mapping:
    - HIGH (≥0.75): Send answer directly without disclaimer
    - MEDIUM (≥0.55): Send answer with support contact disclaimer
    - LOW (≥0.40): Acknowledge query but suggest contacting support
    - REJECT (<0.40): Don't attempt to answer, direct to support immediately
    """
    HIGH = _to_float(os.getenv("CONFIDENCE_HIGH"), 0.75)
    MEDIUM = _to_float(os.getenv("CONFIDENCE_MEDIUM"), 0.55)
    LOW = _to_float(os.getenv("CONFIDENCE_LOW"), 0.18)
    
    @classmethod
    def get_strategy(cls, confidence: float) -> str:
        """
        Determine response strategy based on confidence score
        
        Args:
            confidence: Similarity/confidence score (0.0 to 1.0)
            
        Returns:
            Strategy name: "direct", "cautious", "low_confidence", or "reject"
        """
        if confidence >= cls.HIGH:
            return "direct"
        elif confidence >= cls.MEDIUM:
            return "cautious"
        elif confidence >= cls.LOW:
            return "low_confidence"
        else:
            return "reject"
    
    @classmethod
    def should_answer(cls, confidence: float) -> bool:
        """Returns True if confidence is high enough to attempt an answer"""
        return confidence >= cls.LOW


class Settings:
    # ==================== WhatsApp Configuration ====================
    META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
    META_WHATSAPP_TOKEN = os.getenv("META_WHATSAPP_TOKEN", "")
    WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")
    WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v22.0")
    
    # ==================== Knowledge Base Configuration ====================
    KB_NAMESPACE = os.getenv("KB_NAMESPACE", "default")
    KB_DIR = os.getenv("KB_DIR", "KB")
    
    # Number of top results to retrieve from vector store
    TOP_K = _to_int(os.getenv("TOP_K"), 3)
    
    # Maximum number of KB results to include in LLM context
    MAX_CONTEXT_CHUNKS = _to_int(os.getenv("MAX_CONTEXT_CHUNKS"), 3)
    
    # ==================== LLM Configuration ====================
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    
    # Alias for convenience (falls back to OPENAI_MODEL)
    LLM_MODEL = os.getenv("LLM_MODEL", OPENAI_MODEL)
    
    # LLM temperature (0.0-1.0, lower = more focused/deterministic)
    LLM_TEMPERATURE = _to_float(os.getenv("LLM_TEMPERATURE"), 0.3)
    
    # Maximum tokens for LLM response
    LLM_MAX_TOKENS = _to_int(os.getenv("LLM_MAX_TOKENS"), 300)
    
    # ==================== Embeddings Configuration ====================
    # OpenAI embedding model or HuggingFace model ID
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    
    # For HuggingFace sentence transformers (if not using OpenAI embeddings)
    HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "all-MiniLM-L6-v2")
    
    # ==================== Vector Store Configuration ====================
    # FAISS index paths
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "core/faiss_index")
    FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "index.faiss")
    FAISS_DOCS_FILE = os.getenv("FAISS_DOCS_FILE", "documents.pkl")
    
    # Legacy Postgres (keep for backward compatibility if needed)
    PG_DSN = os.getenv("PG_DSN", "")
    VECTOR_TABLE = os.getenv("VECTOR_TABLE", "kb_chunks")
    
    # ==================== Confidence & Quality Settings ====================
    # Legacy single threshold (deprecated, use ConfidenceThresholds class)
    ANSWER_CONFIDENCE_THRESHOLD = _to_float(
        os.getenv("ANSWER_CONFIDENCE_THRESHOLD"), 
        ConfidenceThresholds.MEDIUM
    )
    
    # Enable/disable guardrails and safety checks
    SAFEGUARD_ENABLE = _to_bool(os.getenv("SAFEGUARD_ENABLE"), True)
    
    # ==================== Response Behavior ====================
    # Support contact information (customize for your organization)
    SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@comemr.com")
    SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "+1-555-COMEMR")
    SUPPORT_DOCS_URL = os.getenv("SUPPORT_DOCS_URL", "https://docs.comemr.com")
    
    # Default fallback message when KB has no answer
    FALLBACK_MESSAGE = os.getenv(
        "FALLBACK_MESSAGE",
        "I don't have enough information to answer that accurately. "
        "Please contact ComEMR Support for assistance."
    )
    
    # Out-of-scope detection keywords (comma-separated in .env)
    OUT_OF_SCOPE_KEYWORDS = [
        kw.strip() 
        for kw in os.getenv(
            "OUT_OF_SCOPE_KEYWORDS", 
            "invest,stock,crypto,bitcoin,trade,recipe,cook,weather,news,politics"
        ).split(",")
    ]
    
    # ==================== Rate Limiting & Performance ====================
    # Maximum concurrent requests (for production deployment)
    MAX_CONCURRENT_REQUESTS = _to_int(os.getenv("MAX_CONCURRENT_REQUESTS"), 10)
    
    # Request timeout in seconds
    REQUEST_TIMEOUT = _to_int(os.getenv("REQUEST_TIMEOUT"), 30)
    
    # ==================== Logging Configuration ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv(
        "LOG_FORMAT", 
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Enable detailed query/response logging (disable in production for privacy)
    LOG_QUERIES = _to_bool(os.getenv("LOG_QUERIES"), True)
    LOG_RESPONSES = _to_bool(os.getenv("LOG_RESPONSES"), False)
    
    # ==================== Feature Flags ====================
    # Enable/disable specific features
    ENABLE_WEB_SEARCH = _to_bool(os.getenv("ENABLE_WEB_SEARCH"), False)
    ENABLE_CONVERSATION_MEMORY = _to_bool(os.getenv("ENABLE_CONVERSATION_MEMORY"), False)
    ENABLE_HUMAN_HANDOFF = _to_bool(os.getenv("ENABLE_HUMAN_HANDOFF"), True)
    
    # ==================== Validation ====================
    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate configuration and return list of warnings/errors
        
        Returns:
            List of validation messages (empty if all valid)
        """
        issues = []
        
        # Critical configs
        if not cls.META_WHATSAPP_TOKEN:
            issues.append("❌ META_WHATSAPP_TOKEN is not set")
        if not cls.WHATSAPP_PHONE_ID:
            issues.append("❌ WHATSAPP_PHONE_ID is not set")
        if not cls.OPENAI_API_KEY:
            issues.append("❌ OPENAI_API_KEY is not set")
        
        # Warnings
        if not os.path.exists(cls.KB_DIR):
            issues.append(f"⚠️  KB directory not found: {cls.KB_DIR}")
        
        faiss_index_path = os.path.join(cls.FAISS_INDEX_DIR, cls.FAISS_INDEX_FILE)
        if not os.path.exists(faiss_index_path):
            issues.append(f"⚠️  FAISS index not found: {faiss_index_path}")
        
        # Threshold validation
        if ConfidenceThresholds.HIGH <= ConfidenceThresholds.MEDIUM:
            issues.append("⚠️  CONFIDENCE_HIGH should be greater than CONFIDENCE_MEDIUM")
        if ConfidenceThresholds.MEDIUM <= ConfidenceThresholds.LOW:
            issues.append("⚠️  CONFIDENCE_MEDIUM should be greater than CONFIDENCE_LOW")
        
        return issues


# Singleton instance
settings = Settings()

# Export confidence thresholds for easy access
confidence_thresholds = ConfidenceThresholds

# Validate on import (optional - comment out if you want manual validation)
_validation_issues = settings.validate()
if _validation_issues:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Configuration issues detected:")
    for issue in _validation_issues:
        logger.warning(f"  {issue}")