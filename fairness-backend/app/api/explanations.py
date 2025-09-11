"""
AI Explanations API endpoints
Provides AI-powered explanations for fairness metrics and bias detection
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from app.models.schemas import ExplanationRequest, ExplanationResponse
from app.services.explanation_service import ExplanationService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize explanation service
explanation_service = ExplanationService()

@router.post("/metric", response_model=ExplanationResponse)
async def explain_metric(request: ExplanationRequest):
    """
    Get AI explanation for fairness metrics
    Expected frontend call: POST /api/explanations/metric
    """
    try:
        if not settings.ENABLE_AI_EXPLANATIONS:
            return ExplanationResponse(
                explanation="AI explanations are currently disabled. Please enable them in the configuration.",
                confidence=0.0,
                sources=[]
            )
        
        explanation = await explanation_service.explain_fairness_metric(
            context=request.context,
            data=request.data,
            question=request.question
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Failed to generate metric explanation: {str(e)}")
        # Return a fallback explanation instead of raising an error
        return ExplanationResponse(
            explanation="Unable to generate AI explanation at this time. Please refer to the documentation for metric definitions.",
            confidence=0.0,
            sources=["Documentation"]
        )

@router.post("/detection", response_model=ExplanationResponse)
async def explain_detection(request: ExplanationRequest):
    """
    Get AI explanation for sensitive feature detection
    Expected frontend call: POST /api/explanations/detection
    """
    try:
        if not settings.ENABLE_AI_EXPLANATIONS:
            return ExplanationResponse(
                explanation="AI explanations are currently disabled. Please enable them in the configuration.",
                confidence=0.0,
                sources=[]
            )
        
        explanation = await explanation_service.explain_bias_detection(
            context=request.context,
            data=request.data,
            question=request.question
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Failed to generate detection explanation: {str(e)}")
        # Return a fallback explanation
        return ExplanationResponse(
            explanation="Unable to generate AI explanation at this time. Sensitive features are detected using statistical tests that measure correlation with the target variable.",
            confidence=0.0,
            sources=["Statistical Analysis"]
        )

@router.post("/mitigation", response_model=ExplanationResponse)
async def explain_mitigation(request: ExplanationRequest):
    """
    Get AI explanation for mitigation strategies
    Expected frontend call: POST /api/explanations/mitigation
    """
    try:
        if not settings.ENABLE_AI_EXPLANATIONS:
            return ExplanationResponse(
                explanation="AI explanations are currently disabled. Please enable them in the configuration.",
                confidence=0.0,
                sources=[]
            )
        
        explanation = await explanation_service.explain_mitigation_strategy(
            context=request.context,
            data=request.data,
            question=request.question
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Failed to generate mitigation explanation: {str(e)}")
        # Return a fallback explanation
        return ExplanationResponse(
            explanation="Unable to generate AI explanation at this time. Mitigation strategies are selected based on bias type, severity, and expected impact on fairness metrics.",
            confidence=0.0,
            sources=["Fairness Literature"]
        )

@router.post("/general", response_model=ExplanationResponse)
async def explain_general(request: ExplanationRequest):
    """
    Get general AI explanation for any fairness-related question
    """
    try:
        if not settings.ENABLE_AI_EXPLANATIONS:
            return ExplanationResponse(
                explanation="AI explanations are currently disabled. Please enable them in the configuration.",
                confidence=0.0,
                sources=[]
            )
        
        explanation = await explanation_service.explain_general(
            context=request.context,
            data=request.data,
            question=request.question
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Failed to generate general explanation: {str(e)}")
        # Return a fallback explanation
        return ExplanationResponse(
            explanation="Unable to generate AI explanation at this time. Please refer to the fairness evaluation documentation for more information.",
            confidence=0.0,
            sources=["Documentation"]
        )

@router.get("/status")
async def get_explanation_status():
    """
    Get the status of AI explanation service
    """
    try:
        status = {
            "enabled": settings.ENABLE_AI_EXPLANATIONS,
            "service": "available" if settings.ENABLE_AI_EXPLANATIONS else "disabled",
            "api_key_configured": bool(settings.OPENAI_API_KEY),
            "supported_types": ["metric", "detection", "mitigation", "general"]
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get explanation status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/configure")
async def configure_explanations(config: Dict[str, Any]):
    """
    Configure AI explanation settings (admin endpoint)
    """
    try:
        # This would typically update configuration
        # For now, return current status
        return {
            "message": "Configuration endpoint - implementation depends on deployment requirements",
            "current_status": settings.ENABLE_AI_EXPLANATIONS
        }
        
    except Exception as e:
        logger.error(f"Failed to configure explanations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")
