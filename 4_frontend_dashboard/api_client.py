import os
import logging
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Configure basics logging for production visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Base IP/URLs from environment variables (e.g., loaded from .env)
# Using Tailscale IPs with appropriate ports, Example: http://100.x.y.z:8000
UBUNTU_NODE_URL = os.getenv("UBUNTU_NODE_URL")
MACBOOK_NODE_URL = os.getenv("MACBOOK_NODE_URL")

# Default timeout in seconds to prevent hanging requests
DEFAULT_TIMEOUT = 10

def _make_request(method: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """
    A helper function to execute requests with error handling and timeouts.
    Returns the JSON response or None if the request fails.
    """
    kwargs.setdefault('timeout', DEFAULT_TIMEOUT)
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout after {kwargs.get('timeout')}s for {url}")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error occurred. Is the tailscale node reachable at {url}?")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred for {url}: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"An unexpected request error occurred for {url}: {e}")
    
    return None


# =====================================================================
# Ubuntu Node Integrations: Math & Data
# =====================================================================

def get_business_data(gstin: str) -> Optional[Dict[str, Any]]:
    """
    Fetches business data associated with a provided GSTIN from the Data Node.
    """
    if not UBUNTU_NODE_URL:
        logger.error("UBUNTU_NODE_URL environment variable is missing.")
        return None
        
    url = f"{UBUNTU_NODE_URL}/api/business_data"  # Modify route to match your FastAPI/Flask app
    params = {"gstin": gstin}
    
    logger.info(f"Fetching business data for GSTIN: {gstin}")
    return _make_request("GET", url, params=params)


def get_ml_score(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Retrieves the machine learning classification score based on input MSME data.
    """
    if not UBUNTU_NODE_URL:
        logger.error("UBUNTU_NODE_URL environment variable is missing.")
        return None
        
    url = f"{UBUNTU_NODE_URL}/api/ml_score"
    logger.info("Retrieving ML score...")
    return _make_request("POST", url, json=data)


def check_fraud(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Checks for potential fraud signals using numerical patterns matching.
    """
    if not UBUNTU_NODE_URL:
        logger.error("UBUNTU_NODE_URL environment variable is missing.")
        return None
        
    url = f"{UBUNTU_NODE_URL}/api/check_fraud"
    logger.info("Executing fraud check analysis...")
    return _make_request("POST", url, json=data)


# =====================================================================
# MacBook M4 Pro (Local) Integrations: AI Reasoning via llama.cpp
# =====================================================================

def extract_text_features(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts semantic text features using LLM hosted on MacBook node.
    """
    if not MACBOOK_NODE_URL:
        logger.error("MACBOOK_NODE_URL environment variable is missing.")
        return None
        
    url = f"{MACBOOK_NODE_URL}/api/extract_features"
    payload = {"text": text}
    
    logger.info("Extracting text features using MacBook LLM node...")
    return _make_request("POST", url, json=payload)


def get_shap_explanation(shap_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generates human-readable SHAP explanations from interpretation data.
    """
    if not MACBOOK_NODE_URL:
        logger.error("MACBOOK_NODE_URL environment variable is missing.")
        return None
        
    url = f"{MACBOOK_NODE_URL}/api/shap_explanation"
    logger.info("Retrieving SHAP AI reasoning explanation...")
    return _make_request("POST", url, json=shap_data)


def Messages(msg: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Stateful conversational chat or messages with contextual reasoning 
    passed to the LLM node.
    """
    if not MACBOOK_NODE_URL:
        logger.error("MACBOOK_NODE_URL environment variable is missing.")
        return None
        
    url = f"{MACBOOK_NODE_URL}/api/messages"
    payload = {
        "msg": msg,
        "context": context
    }
    
    logger.info("Sending message request to MacBook node...")
    return _make_request("POST", url, json=payload)
