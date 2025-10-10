"""
Pydantic models for API request/response validation
Based on frontend data contracts
"""

from pydantic import BaseModel, Field, RootModel, ConfigDict
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class DataType(str, Enum):
    """Supported data types for features"""
    CATEGORICAL = "Categorical"
    NUMERICAL = "Numerical"
    BINARY = "Binary"


class TestType(str, Enum):
    """Statistical test types"""
    CHI_SQUARE = "Chi-Square"
    ANOVA = "ANOVA"
    T_TEST = "T-Test"
    PEARSON = "Pearson"
    HSIC = "HSIC/NOCCO"  # ADD THIS LINE

class EffectSizeLabel(str, Enum):
    """Effect size classifications"""
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"


class SensitivityLevel(str, Enum):
    """Sensitivity level classifications"""
    HIGHLY_SENSITIVE = "Highly Sensitive"
    MODERATELY_SENSITIVE = "Moderately Sensitive"
    LOW_SENSITIVITY = "Low Sensitivity"


class RiskLevel(str, Enum):
    """Risk level classifications"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class MetricStatus(str, Enum):
    """Fairness metric status"""
    BIASED = "Biased"
    FAIR = "Fair"
    WARNING = "Warning"


class OverallRating(str, Enum):
    """Overall fairness rating"""
    POOR_FAIRNESS = "Poor Fairness"
    MODERATE_FAIRNESS = "Moderate Fairness"
    GOOD_FAIRNESS = "Good Fairness"


class MitigationCategory(str, Enum):
    """Mitigation strategy categories"""
    PREPROCESSING = "Preprocessing"
    IN_PROCESSING = "In-processing"
    POST_PROCESSING = "Post-processing"


class AnalysisStatus(str, Enum):
    """Analysis status types"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============ Configuration Models ============

class SensitiveAttribute(BaseModel):
    """Model for sensitive attribute configuration"""
    name: str = Field(..., description="Attribute name")
    label: str = Field(..., description="Human-readable label")
    dataType: DataType = Field(..., description="Data type of the attribute")
    description: str = Field(..., description="Description of the attribute")


class SensitiveAttributesResponse(BaseModel):
    """Response model for sensitive attributes configuration"""
    attributes: List[SensitiveAttribute] = Field(..., description="Available sensitive attributes")


class MitigationOption(BaseModel):
    """Model for mitigation option configuration"""
    name: str = Field(..., description="Option name")
    label: str = Field(..., description="Human-readable label")
    category: MitigationCategory = Field(..., description="Mitigation category")
    description: str = Field(..., description="Description of the option")


class MitigationOptionsResponse(BaseModel):
    """Response model for mitigation options configuration"""
    options: List[MitigationOption] = Field(..., description="Available mitigation options")


# Create alias for StrategyCategory to maintain backward compatibility
StrategyCategory = MitigationCategory


# ============ Validation Models ============

class ValidationResult(BaseModel):
    """Model for validation results"""
    isValid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")


# ============ Request Models ============

class AnalysisStartRequest(BaseModel):
    """Request model for starting analysis"""
    model_config = ConfigDict(protected_namespaces=())
    
    training_dataset_id: str = Field(..., description="ID of uploaded training dataset")
    testing_dataset_id: str = Field(..., description="ID of uploaded testing dataset")
    model_id: str = Field(..., description="ID of uploaded model")
    target_column: str = Field(..., description="Name of target column")
    sensitive_attributes: Optional[List[str]] = Field(default=None, description="Manual sensitive attributes selection")
    auto_detection_enabled: bool = Field(default=True, description="Enable automatic sensitive feature detection")
    significance_level: float = Field(default=0.05, description="Statistical significance level")


class ExplanationRequest(BaseModel):
    """Request model for AI explanations"""
    context: str = Field(..., description="Context for explanation")
    data: Dict[str, Any] = Field(..., description="Data to explain")
    explanation_type: str = Field(..., description="Type of explanation needed")


# ============ Response Models ============

class SensitiveFeature(BaseModel):
    """Model for detected sensitive feature"""
    name: str = Field(..., description="Feature name")
    dataType: DataType = Field(..., description="Data type of feature")
    test: TestType = Field(..., description="Statistical test used")
    pValue: float = Field(..., description="P-value from statistical test")
    effectSize: float = Field(..., description="Effect size")
    effectSizeLabel: EffectSizeLabel = Field(..., description="Effect size classification")
    correlation: float = Field(..., description="Correlation with target")
    sensitivityLevel: SensitivityLevel = Field(..., description="Sensitivity classification")
    groups: List[str] = Field(..., description="Groups within this feature")
    description: str = Field(..., description="Human-readable description")


class SensitiveFeatureSummary(BaseModel):
    """Summary of sensitive feature detection"""
    totalDetected: int = Field(..., description="Total features detected")
    highlySensitiveCount: int = Field(..., description="Highly sensitive features count")
    moderatelySensitiveCount: int = Field(..., description="Moderately sensitive features count")
    riskLevel: RiskLevel = Field(..., description="Overall risk level")


class SensitiveFeatureDetectionResponse(BaseModel):
    """Response model for sensitive feature detection"""
    detectedFeatures: List[SensitiveFeature] = Field(..., description="List of detected sensitive features")
    summary: SensitiveFeatureSummary = Field(..., description="Detection summary")


class FairnessMetric(BaseModel):
    """Model for individual fairness metric"""
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    status: MetricStatus = Field(..., description="Metric status")
    threshold: float = Field(..., description="Threshold for fairness")
    groupRates: Dict[str, float] = Field(..., description="Rates by group")
    description: str = Field(..., description="Metric description")
    tooltip: str = Field(..., description="Additional context")


class AttributeFairnessMetrics(BaseModel):
    """Fairness metrics for a specific attribute"""
    attribute: str = Field(..., description="Attribute name")
    groups: List[str] = Field(..., description="Groups in attribute")
    metrics: List[FairnessMetric] = Field(..., description="Fairness metrics")
    overallScore: float = Field(..., description="Overall fairness score (0-100)")
    overallRating: OverallRating = Field(..., description="Overall fairness rating")
    riskClassification: str = Field(..., description="Risk classification")
    primaryIssues: List[str] = Field(..., description="Primary bias issues")


class FairnessMetricsResponse(RootModel[Dict[str, AttributeFairnessMetrics]]):
    """Response model for fairness metrics"""
    root: Dict[str, AttributeFairnessMetrics] = Field(..., description="Fairness metrics by attribute")


class MitigationStrategy(BaseModel):
    """Model for mitigation strategy"""
    name: str = Field(..., description="Strategy name")
    category: MitigationCategory = Field(..., description="Strategy category")
    fairnessImprovement: float = Field(..., description="Expected fairness improvement percentage")
    accuracyImpact: float = Field(..., description="Expected accuracy impact percentage")
    precisionImpact: float = Field(..., description="Expected precision impact percentage")
    recallImpact: float = Field(..., description="Expected recall impact percentage")
    f1Impact: float = Field(..., description="Expected F1 impact percentage")
    stars: int = Field(..., ge=1, le=3, description="Recommendation rating (1-3 stars)")
    recommendation: str = Field(..., description="Recommendation level")
    description: str = Field(..., description="Strategy description")
    targetAttributes: List[str] = Field(..., description="Target sensitive attributes")


class MitigationStrategiesResponse(BaseModel):
    """Response model for mitigation strategies"""
    strategies: List[MitigationStrategy] = Field(..., description="Available mitigation strategies")


class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    accuracy: float = Field(..., description="Accuracy score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1: float = Field(..., description="F1 score")


class FairnessComparisonMetrics(BaseModel):
    """Fairness metrics comparison"""
    before: List[float] = Field(..., description="Before mitigation scores")
    after: List[float] = Field(..., description="After mitigation scores")
    metrics: List[str] = Field(..., description="Metric names")
    overallScoreBefore: int = Field(..., description="Overall fairness score before mitigation")
    overallScoreAfter: int = Field(..., description="Overall fairness score after mitigation")


class GroupComparison(BaseModel):
    """Group-wise comparison model"""
    attribute: str = Field(..., description="Attribute name")
    groups: List[str] = Field(..., description="Groups in attribute")
    before: Dict[str, float] = Field(..., description="Before mitigation scores by group")
    after: Dict[str, float] = Field(..., description="After mitigation scores by group")
    improvement: Dict[str, float] = Field(..., description="Improvement by group")


class BeforeAfterComparisonResponse(BaseModel):
    """Response model for before/after comparison"""
    strategy: str = Field(..., description="Mitigation strategy name")
    fairnessMetrics: FairnessComparisonMetrics = Field(..., description="Fairness metrics comparison")
    performance: Dict[str, PerformanceMetrics] = Field(..., description="Performance metrics comparison")
    groupComparisons: Dict[str, GroupComparison] = Field(..., description="Group-wise comparisons")


class UploadValidation(BaseModel):
    """Upload validation results"""
    isValid: bool = Field(..., description="Whether upload is valid")
    errors: List[str] = Field(default=[], description="Validation errors")
    warnings: List[str] = Field(default=[], description="Validation warnings")


class DataPreview(BaseModel):
    """Data preview model"""
    rows: int = Field(..., description="Number of rows")
    sampleData: List[Dict[str, Any]] = Field(..., description="Sample data rows")


class UploadResponse(BaseModel):
    """Response model for file uploads"""
    success: bool = Field(..., description="Upload success status")
    filename: str = Field(..., description="Uploaded filename")
    size: int = Field(..., description="File size in bytes")
    upload_id: str = Field(..., description="Unique upload ID")
    columns: Optional[List[str]] = Field(default=None, description="Column names for datasets")
    preview: Optional[DataPreview] = Field(default=None, description="Data preview")
    validation: UploadValidation = Field(..., description="Validation results")


class AnalysisStep(BaseModel):
    """Analysis step model"""
    name: str = Field(..., description="Step name")
    status: AnalysisStatus = Field(..., description="Step status")
    progress: float = Field(..., ge=0, le=100, description="Step progress percentage")


class AnalysisStartResponse(BaseModel):
    """Response model for analysis start"""
    analysisId: str = Field(..., description="Unique analysis ID")
    status: AnalysisStatus = Field(..., description="Analysis status")
    estimatedDuration: int = Field(..., description="Estimated duration in seconds")
    steps: List[AnalysisStep] = Field(..., description="Analysis steps")


class AnalysisStatusResponse(BaseModel):
    """Response model for analysis status"""
    analysisId: str = Field(..., description="Analysis ID")
    status: AnalysisStatus = Field(..., description="Current status")
    progress: float = Field(..., ge=0, le=100, description="Overall progress percentage")
    currentStep: Optional[str] = Field(default=None, description="Current step name")
    steps: List[AnalysisStep] = Field(..., description="Analysis steps")
    results: Optional[Dict[str, Any]] = Field(default=None, description="Available results")


class ConfigResponse(BaseModel):
    """Response model for configuration endpoints"""
    data: List[str] = Field(..., description="Configuration options")


class ExplanationResponse(BaseModel):
    """Response model for AI explanations"""
    explanation: str = Field(..., description="Generated explanation")
    context: str = Field(..., description="Explanation context")
    confidence: float = Field(..., ge=0, le=1, description="Explanation confidence")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
