"""
Analysis API endpoints
Handles fairness analysis lifecycle and results retrieval
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Optional, Dict, Any
import logging
import uuid

from app.models.schemas import (
    AnalysisStartRequest, AnalysisStartResponse, AnalysisStatusResponse,
    SensitiveFeatureDetectionResponse, FairnessMetricsResponse,
    MitigationStrategiesResponse, BeforeAfterComparisonResponse,
    AnalysisStatus, AnalysisStep
)
from app.services.analysis_service import AnalysisService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize analysis service (singleton)
analysis_service = AnalysisService.get_instance()

@router.post("/start", response_model=AnalysisStartResponse)
async def start_analysis(
    request: AnalysisStartRequest,
    background_tasks: BackgroundTasks
):
    """
    Start fairness analysis
    Expected frontend call: POST /api/analysis/start
    """
    try:
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Create analysis steps
        steps = [
            AnalysisStep(name="file_validation", status=AnalysisStatus.PENDING, progress=0),
            AnalysisStep(name="sensitive_feature_detection", status=AnalysisStatus.PENDING, progress=0),
            AnalysisStep(name="fairness_metrics_calculation", status=AnalysisStatus.PENDING, progress=0),
            AnalysisStep(name="mitigation_strategy_generation", status=AnalysisStatus.PENDING, progress=0),
            AnalysisStep(name="before_after_comparison", status=AnalysisStatus.PENDING, progress=0)
        ]
        
        # Store analysis configuration
        analysis_service.create_analysis(analysis_id, request, steps)
        
        # Start background analysis
        background_tasks.add_task(
            analysis_service.run_analysis,
            analysis_id
        )
        
        logger.info(f"Analysis started with ID: {analysis_id}")
        
        return AnalysisStartResponse(
            analysisId=analysis_id,
            status=AnalysisStatus.RUNNING,
            estimatedDuration=settings.MAX_ANALYSIS_TIME,
            steps=steps
        )
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Analysis start failed: {str(e)}")

@router.get("/status/{analysis_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: str):
    """
    Get analysis status and progress
    Expected frontend call: GET /api/analysis/status/{analysis_id}
    """
    try:
        status = analysis_service.get_analysis_status(analysis_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.delete("/{analysis_id}")
async def stop_analysis(analysis_id: str):
    """
    Stop running analysis
    Expected frontend call: DELETE /api/analysis/{analysis_id}
    """
    try:
        success = analysis_service.stop_analysis(analysis_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {"success": True, "message": "Analysis stopped"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis stop failed: {str(e)}")

@router.get("/sensitive-features/{analysis_id}", response_model=SensitiveFeatureDetectionResponse)
async def get_sensitive_features(analysis_id: str):
    """
    Get detected sensitive features
    Expected frontend call: GET /api/analysis/sensitive-features/{analysis_id}
    """
    try:
        results = analysis_service.get_sensitive_features(analysis_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Analysis or results not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get sensitive features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

@router.get("/fairness-metrics/{analysis_id}", response_model=Dict[str, Any])
async def get_fairness_metrics(
    analysis_id: str,
    attribute: Optional[str] = Query(None, description="Specific sensitive attribute to analyze")
):
    """
    Get fairness metrics for all or specific attributes
    Expected frontend calls:
    - GET /api/analysis/fairness-metrics/{analysis_id}
    - GET /api/analysis/fairness-metrics/{analysis_id}?attribute={attr}
    """
    try:
        results = analysis_service.get_fairness_metrics(analysis_id, attribute)
        
        if not results:
            raise HTTPException(status_code=404, detail="Analysis or results not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fairness metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

@router.get("/mitigation-strategies/{analysis_id}", response_model=MitigationStrategiesResponse)
async def get_mitigation_strategies(analysis_id: str):
    """
    Get mitigation strategy recommendations
    Expected frontend call: GET /api/analysis/mitigation-strategies/{analysis_id}
    """
    try:
        results = analysis_service.get_mitigation_strategies(analysis_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Analysis or results not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get mitigation strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

@router.get("/before-after-comparison/{analysis_id}", response_model=BeforeAfterComparisonResponse)
async def get_before_after_comparison(
    analysis_id: str,
    strategy: str = Query(..., description="Mitigation strategy name")
):
    """
    Get before/after comparison for a specific mitigation strategy
    Expected frontend call: GET /api/analysis/before-after-comparison/{analysis_id}?strategy={s}
    """
    try:
        logger.info(f"Fetching before/after comparison for analysis: {analysis_id}, strategy: {strategy}")
        results = analysis_service.get_before_after_comparison(analysis_id, strategy)
        
        if not results:
            logger.warning(f"No before/after comparison found for analysis: {analysis_id}, strategy: {strategy}")
            raise HTTPException(status_code=404, detail="Analysis, strategy, or results not found")
        
        # Log available group comparisons for debugging
        if hasattr(results, 'groupComparisons') and results.groupComparisons:
            available_attrs = list(results.groupComparisons.keys())
            logger.info(f"Available group comparison attributes: {available_attrs}")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get before/after comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

@router.get("/results/{analysis_id}")
async def get_all_results(analysis_id: str):
    """
    Get all analysis results in one call (convenience endpoint)
    """
    try:
        results = analysis_service.get_all_results(analysis_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get all results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

@router.get("/list")
async def list_analyses():
    """
    List all analyses (for debugging/admin purposes)
    """
    try:
        analyses = analysis_service.list_all_analyses()
        return {"analyses": analyses}
        
    except Exception as e:
        logger.error(f"Failed to list analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"List retrieval failed: {str(e)}")


@router.get("/test-mitigation-calculation", response_model=BeforeAfterComparisonResponse)
async def test_mitigation_calculation(
    strategy: str = Query(default="Reweighing", description="Mitigation strategy to test")
):
    """
    Test endpoint to verify that overall fairness scores are calculated correctly
    This simulates the before/after comparison with mock data but real calculations
    """
    try:
        logger.info(f"Testing mitigation calculation for strategy: {strategy}")
        
        # Create mock fairness results and sensitive attributes for testing
        mock_fairness_results = {
            "gender": {
                "statistical_parity": 0.42,
                "disparate_impact": 0.38,
                "equal_opportunity": 0.45,
                "equalized_odds": 0.35,
                "calibration": 0.48
            }
        }
        
        mock_sensitive_attrs = ["gender", "race", "education_years", "work_experience", "skill_score", "previous_salary", "age"]
        
        # Create a mock strategy object with the expected attributes
        class MockStrategy:
            def __init__(self, name: str):
                self.name = name
                self.fairnessImprovement = 28.4  # Mock improvement value
                self.accuracyImpact = -1.8       # Mock accuracy impact
                self.precisionImpact = -1.6      # Mock precision impact
                self.recallImpact = -1.6         # Mock recall impact
                self.f1Impact = -1.7             # Mock F1 impact
        
        mock_strategy_obj = MockStrategy(strategy)
        
        # Create realistic mock sensitive features data that mimics what bias detection would produce
        mock_sensitive_features = {
            "detectedFeatures": [
                {"name": "gender", "groups": ["0", "1"]},  # Binary encoded
                {"name": "race", "groups": ["0", "1", "2", "3"]},  # Multi-class encoded
                {"name": "education_years", "groups": ["12.0", "14.0", "16.0", "18.0", "20.0"]},  # Actual years
                {"name": "work_experience", "groups": ["0.0", "3.5", "7.2", "12.8", "18.5"]},  # Actual years
                {"name": "skill_score", "groups": ["45.2", "58.7", "72.3", "85.1", "94.6"]},  # Actual scores
                {"name": "previous_salary", "groups": ["45000", "65000", "85000", "105000", "125000"]},  # Actual salaries
                {"name": "age", "groups": ["25", "35", "45", "55", "65"]}  # Age ranges
            ]
        }
        
        # Create a mock analysis result using the same logic as the real endpoint
        mock_analysis_result = analysis_service._simulate_before_after_comparison(
            mock_strategy_obj, mock_fairness_results, mock_sensitive_attrs, mock_sensitive_features
        )
        
        logger.info(f"Test calculation result: {mock_analysis_result}")
        
        return mock_analysis_result
        
    except Exception as e:
        logger.error(f"Failed to test mitigation calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test calculation failed: {str(e)}")


@router.post("/refresh-comparisons/{analysis_id}")
async def refresh_before_after_comparisons(analysis_id: str):
    """
    Refresh/regenerate before-after comparisons for an existing analysis
    This will update the cached comparisons with the latest logic
    """
    try:
        logger.info(f"Refreshing before/after comparisons for analysis: {analysis_id}")
        
        analysis = analysis_service.analyses.get(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Get the existing mitigation strategies and sensitive attributes
        mitigation_strategies = analysis["results"].get("mitigation_strategies")
        if not mitigation_strategies:
            raise HTTPException(status_code=400, detail="No mitigation strategies found for this analysis")
        
        # Get sensitive features from the analysis
        sensitive_features = analysis["results"].get("sensitive_features")
        if not sensitive_features:
            raise HTTPException(status_code=400, detail="No sensitive features found for this analysis")
        
        sensitive_attrs = [feature["name"] for feature in sensitive_features["detectedFeatures"]]
        
        # Get fairness results
        fairness_results = analysis["results"].get("fairness_metrics", {})
        
        # Regenerate before/after comparisons for each strategy
        before_after_results = {}
        for strategy in mitigation_strategies.strategies:
            comparison = analysis_service._simulate_before_after_comparison(
                strategy, fairness_results, sensitive_attrs, sensitive_features
            )
            before_after_results[strategy.name] = comparison
        
        # Update the stored analysis with new comparisons
        analysis["results"]["before_after_comparisons"] = before_after_results
        
        logger.info(f"Successfully refreshed comparisons for {len(before_after_results)} strategies")
        
        return {
            "message": f"Successfully refreshed before/after comparisons for {len(before_after_results)} strategies",
            "strategies": list(before_after_results.keys()),
            "available_attributes": sensitive_attrs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh before/after comparisons: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")
