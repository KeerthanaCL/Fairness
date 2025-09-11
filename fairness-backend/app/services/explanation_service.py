"""
AI explanation service for generating human-readable explanations
Optional service that provides AI-powered explanations for fairness concepts
"""

from typing import Dict, Any, Optional
import logging

from app.models.schemas import ExplanationResponse
from app.core.config import settings

logger = logging.getLogger(__name__)


class ExplanationService:
    """Provides AI-powered explanations for fairness analysis results"""
    
    def __init__(self):
        self.enabled = settings.ENABLE_AI_EXPLANATIONS
        self.api_key = settings.OPENAI_API_KEY
        
    async def explain_fairness_metric(
        self, 
        context: str, 
        data: Dict[str, Any], 
        question: Optional[str] = None
    ) -> ExplanationResponse:
        """Generate explanation for fairness metrics"""
        
        if not self.enabled:
            return self._get_fallback_metric_explanation(context, data)
        
        try:
            # This would integrate with OpenAI or other LLM services
            # For now, return rule-based explanations
            return self._generate_rule_based_metric_explanation(context, data, question)
            
        except Exception as e:
            logger.error(f"Failed to generate AI explanation: {str(e)}")
            return self._get_fallback_metric_explanation(context, data)
    
    async def explain_bias_detection(
        self, 
        context: str, 
        data: Dict[str, Any], 
        question: Optional[str] = None
    ) -> ExplanationResponse:
        """Generate explanation for bias detection results"""
        
        if not self.enabled:
            return self._get_fallback_detection_explanation(context, data)
        
        try:
            return self._generate_rule_based_detection_explanation(context, data, question)
            
        except Exception as e:
            logger.error(f"Failed to generate AI explanation: {str(e)}")
            return self._get_fallback_detection_explanation(context, data)
    
    async def explain_mitigation_strategy(
        self, 
        context: str, 
        data: Dict[str, Any], 
        question: Optional[str] = None
    ) -> ExplanationResponse:
        """Generate explanation for mitigation strategies"""
        
        if not self.enabled:
            return self._get_fallback_mitigation_explanation(context, data)
        
        try:
            return self._generate_rule_based_mitigation_explanation(context, data, question)
            
        except Exception as e:
            logger.error(f"Failed to generate AI explanation: {str(e)}")
            return self._get_fallback_mitigation_explanation(context, data)
    
    async def explain_general(
        self, 
        context: str, 
        data: Dict[str, Any], 
        question: Optional[str] = None
    ) -> ExplanationResponse:
        """Generate general explanation for fairness concepts"""
        
        if not self.enabled:
            return self._get_fallback_general_explanation(context, data)
        
        try:
            return self._generate_rule_based_general_explanation(context, data, question)
            
        except Exception as e:
            logger.error(f"Failed to generate AI explanation: {str(e)}")
            return self._get_fallback_general_explanation(context, data)
    
    def _generate_rule_based_metric_explanation(
        self, 
        context: str, 
        data: Dict[str, Any], 
        question: Optional[str]
    ) -> ExplanationResponse:
        """Generate rule-based explanation for metrics"""
        
        metric_name = data.get("metric_name", "unknown")
        metric_value = data.get("metric_value", 0)
        status = data.get("status", "unknown")
        
        explanations = {
            "Statistical Parity": f"Statistical Parity measures the difference in positive prediction rates between different groups. "
                                f"A value of {metric_value:.3f} indicates {'bias' if abs(metric_value) > 0.1 else 'fairness'} "
                                f"in how often the model predicts positive outcomes for different groups. "
                                f"Values closer to 0 indicate more fairness.",
            
            "Disparate Impact": f"Disparate Impact is the ratio of positive prediction rates between groups. "
                              f"A value of {metric_value:.3f} {'is within' if 0.8 <= metric_value <= 1.25 else 'violates'} "
                              f"the '80% rule' used in employment discrimination law. "
                              f"Values between 0.8 and 1.25 are generally considered fair.",
            
            "Equal Opportunity": f"Equal Opportunity measures the difference in true positive rates between groups. "
                               f"A value of {metric_value:.3f} indicates {'bias' if abs(metric_value) > 0.1 else 'fairness'} "
                               f"in how often the model correctly identifies positive cases across groups. "
                               f"Values closer to 0 indicate equal opportunity.",
            
            "Equalized Odds": f"Equalized Odds ensures both true positive and false positive rates are equal across groups. "
                            f"A value of {metric_value:.3f} indicates {'bias' if abs(metric_value) > 0.1 else 'fairness'} "
                            f"in the model's error rates across different groups. "
                            f"Lower values indicate more equalized performance."
        }
        
        explanation = explanations.get(metric_name, 
            f"This metric measures fairness with a value of {metric_value:.3f}. "
            f"The status is '{status}', indicating the level of bias detected.")
        
        return ExplanationResponse(
            explanation=explanation,
            confidence=0.8,
            sources=["Fairness ML Literature", "Statistical Analysis"]
        )
    
    def _generate_rule_based_detection_explanation(
        self, 
        context: str, 
        data: Dict[str, Any], 
        question: Optional[str]
    ) -> ExplanationResponse:
        """Generate rule-based explanation for bias detection"""
        
        feature_name = data.get("feature_name", "unknown")
        test_type = data.get("test_type", "unknown")
        p_value = data.get("p_value", 1.0)
        sensitivity_level = data.get("sensitivity_level", "unknown")
        
        test_explanations = {
            "Chi-Square": f"The Chi-Square test was used to detect bias in the categorical feature '{feature_name}'. "
                         f"With a p-value of {p_value:.4f}, this indicates a "
                         f"{'significant' if p_value < 0.05 else 'non-significant'} statistical relationship "
                         f"with the target variable.",
            
            "ANOVA": f"Analysis of Variance (ANOVA) was used to test the categorical feature '{feature_name}' "
                    f"against the numerical target. A p-value of {p_value:.4f} "
                    f"{'suggests' if p_value < 0.05 else 'does not suggest'} significant differences "
                    f"between groups.",
            
            "Pearson Correlation": f"Pearson correlation analysis was performed on the numerical feature '{feature_name}'. "
                                 f"The p-value of {p_value:.4f} indicates a "
                                 f"{'significant' if p_value < 0.05 else 'non-significant'} linear relationship "
                                 f"with the target variable."
        }
        
        base_explanation = test_explanations.get(test_type, 
            f"Statistical testing was performed on feature '{feature_name}' with p-value {p_value:.4f}.")
        
        sensitivity_explanation = f" This feature is classified as '{sensitivity_level}' " \
                                f"for potential bias, meaning it {'requires careful attention' if 'High' in sensitivity_level else 'should be monitored'}."
        
        explanation = base_explanation + sensitivity_explanation
        
        return ExplanationResponse(
            explanation=explanation,
            confidence=0.85,
            sources=["Statistical Testing", "Bias Detection Algorithms"]
        )
    
    def _generate_rule_based_mitigation_explanation(
        self, 
        context: str, 
        data: Dict[str, Any], 
        question: Optional[str]
    ) -> ExplanationResponse:
        """Generate rule-based explanation for mitigation strategies"""
        
        strategy_name = data.get("strategy_name", "unknown")
        category = data.get("category", "unknown")
        fairness_improvement = data.get("fairness_improvement", 0)
        accuracy_impact = data.get("accuracy_impact", 0)
        
        strategy_explanations = {
            "Reweighing": "Reweighing is a preprocessing technique that adjusts the weights of training examples "
                         "to reduce bias. It gives higher weights to underrepresented groups and lower weights "
                         "to overrepresented groups, helping to balance the training data.",
            
            "Adversarial Debiasing": "Adversarial Debiasing uses adversarial learning during model training. "
                                   "It trains two neural networks simultaneously: one to make predictions and "
                                   "another to detect sensitive attributes from predictions. This forces the "
                                   "main model to make predictions that don't reveal sensitive information.",
            
            "Calibrated Equalized Odds": "Calibrated Equalized Odds is a post-processing technique that adjusts "
                                       "the decision thresholds for different groups to achieve equal true positive "
                                       "and false positive rates across groups."
        }
        
        base_explanation = strategy_explanations.get(strategy_name,
            f"This {category.lower()} strategy works to reduce bias in machine learning models.")
        
        impact_explanation = f" This strategy is expected to improve fairness by {fairness_improvement:.1f}% " \
                           f"with an accuracy impact of {accuracy_impact:+.1f}%."
        
        explanation = base_explanation + impact_explanation
        
        return ExplanationResponse(
            explanation=explanation,
            confidence=0.8,
            sources=["Fairness ML Research", "Mitigation Algorithms"]
        )
    
    def _generate_rule_based_general_explanation(
        self, 
        context: str, 
        data: Dict[str, Any], 
        question: Optional[str]
    ) -> ExplanationResponse:
        """Generate rule-based general explanation"""
        
        if question:
            question_lower = question.lower()
            
            if "fairness" in question_lower:
                explanation = "Fairness in machine learning refers to the absence of bias or discrimination " \
                            "in model predictions across different groups. It ensures that the model treats " \
                            "all individuals or groups equitably, regardless of sensitive attributes like " \
                            "race, gender, age, or other protected characteristics."
            
            elif "bias" in question_lower:
                explanation = "Bias in machine learning occurs when a model systematically discriminates " \
                            "against certain groups or individuals. This can happen due to biased training " \
                            "data, biased feature selection, or algorithms that inadvertently learn to " \
                            "discriminate based on sensitive attributes."
            
            elif "mitigation" in question_lower:
                explanation = "Bias mitigation involves techniques to reduce or eliminate unfair discrimination " \
                            "in machine learning models. These techniques can be applied at three stages: " \
                            "preprocessing (modify data), in-processing (modify algorithm), or " \
                            "post-processing (modify predictions)."
            
            else:
                explanation = "This platform analyzes machine learning models for fairness and bias. " \
                            "It detects potential discrimination, measures fairness metrics, and " \
                            "recommends strategies to improve model fairness while maintaining performance."
        else:
            explanation = "Fairness evaluation is crucial for responsible AI deployment. This platform " \
                        "helps identify and mitigate bias in machine learning models to ensure " \
                        "equitable treatment across different demographic groups."
        
        return ExplanationResponse(
            explanation=explanation,
            confidence=0.7,
            sources=["Fairness ML Literature", "General Knowledge"]
        )
    
    # Fallback explanations when AI is disabled
    def _get_fallback_metric_explanation(self, context: str, data: Dict[str, Any]) -> ExplanationResponse:
        return ExplanationResponse(
            explanation="AI explanations are currently disabled. Please refer to the fairness metrics documentation for detailed explanations of each metric and their interpretations.",
            confidence=0.0,
            sources=["Documentation"]
        )
    
    def _get_fallback_detection_explanation(self, context: str, data: Dict[str, Any]) -> ExplanationResponse:
        return ExplanationResponse(
            explanation="AI explanations are currently disabled. Sensitive features are detected using statistical tests that measure the relationship between features and the target variable.",
            confidence=0.0,
            sources=["Statistical Analysis"]
        )
    
    def _get_fallback_mitigation_explanation(self, context: str, data: Dict[str, Any]) -> ExplanationResponse:
        return ExplanationResponse(
            explanation="AI explanations are currently disabled. Mitigation strategies are recommended based on the type and severity of detected bias. Please refer to the documentation for detailed strategy descriptions.",
            confidence=0.0,
            sources=["Documentation"]
        )
    
    def _get_fallback_general_explanation(self, context: str, data: Dict[str, Any]) -> ExplanationResponse:
        return ExplanationResponse(
            explanation="AI explanations are currently disabled. Please refer to the fairness evaluation platform documentation for comprehensive information about bias detection and mitigation.",
            confidence=0.0,
            sources=["Documentation"]
        )
