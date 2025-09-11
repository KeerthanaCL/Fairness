"""
Configuration API endpoints
Provides configuration options for the frontend
"""

from fastapi import APIRouter, HTTPException
from typing import List
import logging

from app.models.schemas import (
    SensitiveAttributesResponse, MitigationOptionsResponse,
    SensitiveAttribute, MitigationOption, DataType, StrategyCategory
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/sensitive-attributes", response_model=SensitiveAttributesResponse)
async def get_sensitive_attributes():
    """
    Get available sensitive attributes for configuration
    Expected frontend call: GET /api/config/sensitive-attributes
    """
    try:
        # Define common sensitive attributes
        attributes = [
            SensitiveAttribute(
                name="gender",
                label="Gender",
                dataType=DataType.CATEGORICAL,
                description="Gender-based discrimination analysis"
            ),
            SensitiveAttribute(
                name="age",
                label="Age",
                dataType=DataType.NUMERICAL,
                description="Age-based bias detection"
            ),
            SensitiveAttribute(
                name="age_group",
                label="Age Group",
                dataType=DataType.CATEGORICAL,
                description="Age group categorization for fairness analysis"
            ),
            SensitiveAttribute(
                name="race",
                label="Race/Ethnicity",
                dataType=DataType.CATEGORICAL,
                description="Racial bias and discrimination analysis"
            ),
            SensitiveAttribute(
                name="ethnicity",
                label="Ethnicity",
                dataType=DataType.CATEGORICAL,
                description="Ethnic discrimination detection"
            ),
            SensitiveAttribute(
                name="religion",
                label="Religion",
                dataType=DataType.CATEGORICAL,
                description="Religious bias analysis"
            ),
            SensitiveAttribute(
                name="nationality",
                label="Nationality",
                dataType=DataType.CATEGORICAL,
                description="Nationality-based discrimination detection"
            ),
            SensitiveAttribute(
                name="marital_status",
                label="Marital Status",
                dataType=DataType.CATEGORICAL,
                description="Marital status bias analysis"
            ),
            SensitiveAttribute(
                name="disability",
                label="Disability Status",
                dataType=DataType.BINARY,
                description="Disability-based discrimination analysis"
            ),
            SensitiveAttribute(
                name="sexual_orientation",
                label="Sexual Orientation",
                dataType=DataType.CATEGORICAL,
                description="Sexual orientation bias detection"
            ),
            SensitiveAttribute(
                name="income",
                label="Income Level",
                dataType=DataType.NUMERICAL,
                description="Income-based bias analysis"
            ),
            SensitiveAttribute(
                name="education",
                label="Education Level",
                dataType=DataType.CATEGORICAL,
                description="Educational background bias detection"
            ),
            SensitiveAttribute(
                name="location",
                label="Geographic Location",
                dataType=DataType.CATEGORICAL,
                description="Location-based discrimination analysis"
            ),
            SensitiveAttribute(
                name="zip_code",
                label="ZIP Code",
                dataType=DataType.CATEGORICAL,
                description="ZIP code-based bias detection"
            ),
            SensitiveAttribute(
                name="family_size",
                label="Family Size",
                dataType=DataType.NUMERICAL,
                description="Family size bias analysis"
            )
        ]
        
        return SensitiveAttributesResponse(attributes=attributes)
        
    except Exception as e:
        logger.error(f"Failed to get sensitive attributes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration retrieval failed: {str(e)}")

@router.get("/mitigation-options", response_model=MitigationOptionsResponse)
async def get_mitigation_options():
    """
    Get available mitigation strategy options
    Expected frontend call: GET /api/config/mitigation-options
    """
    try:
        # Define available mitigation strategies
        options = [
            # Preprocessing strategies
            MitigationOption(
                name="Reweighing",
                category=StrategyCategory.PREPROCESSING,
                description="Adjust instance weights to reduce bias in training data",
                parameters={
                    "unprivileged_groups": "Groups to give higher weights",
                    "privileged_groups": "Groups to give lower weights"
                }
            ),
            MitigationOption(
                name="Disparate Impact Remover",
                category=StrategyCategory.PREPROCESSING,
                description="Remove disparate impact by editing feature values",
                parameters={
                    "repair_level": "Level of repair (0.0 to 1.0)"
                }
            ),
            MitigationOption(
                name="Learning Fair Representations",
                category=StrategyCategory.PREPROCESSING,
                description="Learn fair data representations that obscure protected attributes",
                parameters={
                    "k": "Number of prototypes",
                    "Ax": "Fairness constraint parameter",
                    "Ay": "Accuracy constraint parameter"
                }
            ),
            MitigationOption(
                name="Optimized Preprocessing",
                category=StrategyCategory.PREPROCESSING,
                description="Optimize preprocessing to improve fairness metrics",
                parameters={
                    "optim_options": "Optimization parameters",
                    "distortion_fun": "Distortion function"
                }
            ),
            
            # In-processing strategies
            MitigationOption(
                name="Adversarial Debiasing",
                category=StrategyCategory.IN_PROCESSING,
                description="Use adversarial learning to reduce bias during training",
                parameters={
                    "adversary_loss_weight": "Weight for adversarial loss",
                    "num_epochs": "Number of training epochs"
                }
            ),
            MitigationOption(
                name="Fair Classification",
                category=StrategyCategory.IN_PROCESSING,
                description="Incorporate fairness constraints directly into the learning algorithm",
                parameters={
                    "fairness_constraint": "Type of fairness constraint",
                    "constraint_weight": "Weight for fairness constraint"
                }
            ),
            MitigationOption(
                name="Grid Search Reduction",
                category=StrategyCategory.IN_PROCESSING,
                description="Reduce fairness problems via grid search over classifiers",
                parameters={
                    "constraints": "Fairness constraints to satisfy",
                    "grid_size": "Size of the search grid"
                }
            ),
            MitigationOption(
                name="Exponentiated Gradient",
                category=StrategyCategory.IN_PROCESSING,
                description="Optimize fairness constraints using exponentiated gradient",
                parameters={
                    "constraints": "Fairness constraints",
                    "eps": "Tolerance for constraint violation"
                }
            ),
            
            # Post-processing strategies
            MitigationOption(
                name="Calibrated Equalized Odds",
                category=StrategyCategory.POST_PROCESSING,
                description="Adjust predictions to satisfy equalized odds",
                parameters={
                    "cost_constraint": "Cost constraint type",
                    "random_seed": "Random seed for reproducibility"
                }
            ),
            MitigationOption(
                name="Reject Option Classification",
                category=StrategyCategory.POST_PROCESSING,
                description="Improve fairness by rejecting uncertain predictions",
                parameters={
                    "low_class_thresh": "Low class threshold",
                    "high_class_thresh": "High class threshold",
                    "num_class_thresh": "Number of threshold values"
                }
            ),
            MitigationOption(
                name="Equalized Odds Postprocessing",
                category=StrategyCategory.POST_PROCESSING,
                description="Post-process predictions to achieve equalized odds",
                parameters={
                    "unprivileged_groups": "Unprivileged groups",
                    "privileged_groups": "Privileged groups"
                }
            ),
            MitigationOption(
                name="Threshold Optimizer",
                category=StrategyCategory.POST_PROCESSING,
                description="Optimize decision thresholds for different groups",
                parameters={
                    "fairness_metric": "Fairness metric to optimize",
                    "threshold_range": "Range of thresholds to consider"
                }
            )
        ]
        
        return MitigationOptionsResponse(options=options)
        
    except Exception as e:
        logger.error(f"Failed to get mitigation options: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration retrieval failed: {str(e)}")

@router.get("/fairness-metrics")
async def get_fairness_metrics_info():
    """
    Get information about available fairness metrics
    """
    try:
        metrics_info = {
            "statistical_parity": {
                "name": "Statistical Parity",
                "description": "Measures difference in positive prediction rates between groups",
                "formula": "P(Y=1|A=unprivileged) - P(Y=1|A=privileged)",
                "ideal_value": 0.0,
                "tolerance": 0.1
            },
            "disparate_impact": {
                "name": "Disparate Impact",
                "description": "Ratio of positive prediction rates between groups",
                "formula": "P(Y=1|A=unprivileged) / P(Y=1|A=privileged)",
                "ideal_value": 1.0,
                "tolerance": [0.8, 1.2]
            },
            "equal_opportunity": {
                "name": "Equal Opportunity",
                "description": "Difference in true positive rates between groups",
                "formula": "TPR(unprivileged) - TPR(privileged)",
                "ideal_value": 0.0,
                "tolerance": 0.1
            },
            "equalized_odds": {
                "name": "Equalized Odds",
                "description": "Maximum difference in TPR and FPR between groups",
                "formula": "max(|TPR_diff|, |FPR_diff|)",
                "ideal_value": 0.0,
                "tolerance": 0.1
            },
            "calibration": {
                "name": "Calibration",
                "description": "Difference in positive predictive values between groups",
                "formula": "PPV(unprivileged) - PPV(privileged)",
                "ideal_value": 0.0,
                "tolerance": 0.1
            }
        }
        
        return {"metrics": metrics_info}
        
    except Exception as e:
        logger.error(f"Failed to get fairness metrics info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration retrieval failed: {str(e)}")

@router.get("/data-types")
async def get_supported_data_types():
    """
    Get supported data types for features
    """
    try:
        data_types = {
            "categorical": {
                "name": "Categorical",
                "description": "Discrete categories (e.g., Male/Female, Red/Blue/Green)",
                "examples": ["gender", "race", "education_level"]
            },
            "numerical": {
                "name": "Numerical",
                "description": "Continuous or discrete numeric values",
                "examples": ["age", "income", "score"]
            },
            "binary": {
                "name": "Binary",
                "description": "Two-value categorical (e.g., Yes/No, True/False)",
                "examples": ["married", "has_disability", "is_citizen"]
            }
        }
        
        return {"data_types": data_types}
        
    except Exception as e:
        logger.error(f"Failed to get data types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration retrieval failed: {str(e)}")
