"""
Fairness metrics calculation module adapted from backend_old
Calculates comprehensive fairness metrics for sensitive features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import logging

from app.models.schemas import (
    FairnessMetricsResponse, AttributeFairnessMetrics, FairnessMetric,
    MetricStatus, OverallRating
)

logger = logging.getLogger(__name__)


class FairnessMetricsCalculator:
    """Enhanced fairness metrics calculator with comprehensive bias analysis"""
    
    def __init__(self):
        self.metrics_results = {}
        
        # Fairness thresholds
        self.thresholds = {
            'statistical_parity': 0.10,      # 10% difference acceptable
            'disparate_impact': 0.80,        # 80% rule
            'equal_opportunity': 0.10,       # 10% difference in TPR
            'equalized_odds': 0.10,          # 10% difference in TPR and FPR
            'calibration': 0.10,             # 10% difference in calibration
            'generalized_entropy_index': 0.30 # 30% entropy index
        }
    
    def calculate_all_metrics(
        self, 
        df: pd.DataFrame, 
        model: Any, 
        sensitive_features: List[str], 
        target_column: str, 
        feature_columns: List[str],
        attribute_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate fairness metrics for all or specific sensitive features
        """
        try:
            # Get model predictions
            X = df[feature_columns]
            y_true = df[target_column]
            
            y_pred = model.predict(X)
            y_pred_proba = self._get_prediction_probabilities(model, X)
            
            results = {}
            
            # Filter to specific attribute if requested
            features_to_analyze = [attribute_filter] if attribute_filter and attribute_filter in sensitive_features else sensitive_features
            
            for sensitive_feature in features_to_analyze:
                try:
                    feature_metrics = self._calculate_feature_metrics(
                        df, sensitive_feature, y_true, y_pred, y_pred_proba, target_column
                    )
                    results[sensitive_feature] = feature_metrics
                    
                except Exception as e:
                    logger.warning(f"Error calculating metrics for {sensitive_feature}: {str(e)}")
                    continue
            
            self.metrics_results = results
            logger.info(f"Fairness metrics calculated for {len(results)} sensitive features")
            
            return results
            
        except Exception as e:
            logger.error(f"Fairness metrics calculation failed: {str(e)}")
            raise
    
    def _get_prediction_probabilities(self, model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities if available"""
        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X)
                # Convert to probabilities using sigmoid for binary classification
                if scores.ndim == 1:
                    return 1 / (1 + np.exp(-scores))
                return scores
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {str(e)}")
            return None
    
    def _calculate_feature_metrics(
        self, 
        df: pd.DataFrame,
        sensitive_feature: str, 
        y_true: pd.Series, 
        y_pred: pd.Series,
        y_pred_proba: Optional[np.ndarray],
        target_column: str
    ) -> AttributeFairnessMetrics:
        """
        Calculate all fairness metrics for a single sensitive feature
        """
        sensitive_attribute = df[sensitive_feature]
        groups = sorted(sensitive_attribute.dropna().unique().astype(str))
        
        if len(groups) < 2:
            # Return default metrics for insufficient groups
            return self._create_default_metrics(sensitive_feature, groups)
        
        # Calculate individual metrics
        metrics = []
        
        # 1. Statistical Parity (Demographic Parity)
        sp_metric = self._calculate_statistical_parity(sensitive_attribute, y_pred, groups)
        metrics.append(sp_metric)
        
        # 2. Disparate Impact
        di_metric = self._calculate_disparate_impact(sensitive_attribute, y_pred, groups)
        metrics.append(di_metric)
        
        # 3. Equal Opportunity
        eo_metric = self._calculate_equal_opportunity(sensitive_attribute, y_true, y_pred, groups)
        metrics.append(eo_metric)
        
        # 4. Equalized Odds
        eod_metric = self._calculate_equalized_odds(sensitive_attribute, y_true, y_pred, groups)
        metrics.append(eod_metric)
        
        # 5. Calibration (if probabilities available)
        if y_pred_proba is not None:
            cal_metric = self._calculate_calibration(sensitive_attribute, y_true, y_pred_proba, groups)
            metrics.append(cal_metric)
        
        # 6. Generalized Entropy Index
        gei_metric = self._calculate_generalized_entropy_index(sensitive_attribute, y_pred, groups)
        metrics.append(gei_metric)
        
        # Calculate overall score and rating
        overall_score = self._calculate_overall_score(metrics)
        overall_rating = self._determine_overall_rating(overall_score)
        risk_classification = self._determine_risk_classification(metrics)
        primary_issues = self._identify_primary_issues(metrics, sensitive_feature)
        
        return AttributeFairnessMetrics(
            attribute=sensitive_feature,
            groups=groups,
            metrics=metrics,
            overallScore=overall_score,
            overallRating=overall_rating,
            riskClassification=risk_classification,
            primaryIssues=primary_issues
        )
    
    def _calculate_statistical_parity(self, sensitive_attr: pd.Series, y_pred: pd.Series, groups: List[str]) -> FairnessMetric:
        """Calculate Statistical Parity (Demographic Parity)"""
        try:
            group_rates = {}
            for group in groups:
                group_mask = sensitive_attr.astype(str) == group
                if group_mask.sum() > 0:
                    positive_rate = y_pred[group_mask].mean()
                    group_rates[group] = round(positive_rate, 3)
                else:
                    group_rates[group] = 0.0
            
            # Calculate maximum difference between groups
            if len(group_rates) > 1:
                rates = list(group_rates.values())
                max_diff = max(rates) - min(rates)
            else:
                max_diff = 0.0
            
            status = MetricStatus.FAIR if max_diff <= self.thresholds['statistical_parity'] else MetricStatus.BIASED
            
            return FairnessMetric(
                name="Statistical Parity",
                value=round(max_diff, 3),
                status=status,
                threshold=self.thresholds['statistical_parity'],
                groupRates=group_rates,
                description="Measures whether positive prediction rates are equal across groups",
                tooltip="Lower values indicate better fairness. Values > 0.10 suggest bias."
            )
            
        except Exception as e:
            logger.warning(f"Statistical parity calculation failed: {str(e)}")
            return self._create_error_metric("Statistical Parity", str(e))
    
    def _calculate_disparate_impact(self, sensitive_attr: pd.Series, y_pred: pd.Series, groups: List[str]) -> FairnessMetric:
        """Calculate Disparate Impact (80% rule)"""
        try:
            group_rates = {}
            for group in groups:
                group_mask = sensitive_attr.astype(str) == group
                if group_mask.sum() > 0:
                    positive_rate = y_pred[group_mask].mean()
                    group_rates[group] = round(positive_rate, 3)
                else:
                    group_rates[group] = 0.0
            
            # Calculate disparate impact ratio (min rate / max rate)
            if len(group_rates) > 1:
                rates = [rate for rate in group_rates.values() if rate > 0]
                if len(rates) > 1:
                    di_ratio = min(rates) / max(rates)
                else:
                    di_ratio = 1.0
            else:
                di_ratio = 1.0
            
            status = MetricStatus.FAIR if di_ratio >= self.thresholds['disparate_impact'] else MetricStatus.BIASED
            
            return FairnessMetric(
                name="Disparate Impact",
                value=round(di_ratio, 3),
                status=status,
                threshold=self.thresholds['disparate_impact'],
                groupRates=group_rates,
                description="Ratio of selection rates between groups (80% rule)",
                tooltip="Values >= 0.80 are considered fair. Lower values indicate disparate impact."
            )
            
        except Exception as e:
            logger.warning(f"Disparate impact calculation failed: {str(e)}")
            return self._create_error_metric("Disparate Impact", str(e))
    
    def _calculate_equal_opportunity(self, sensitive_attr: pd.Series, y_true: pd.Series, y_pred: pd.Series, groups: List[str]) -> FairnessMetric:
        """Calculate Equal Opportunity (True Positive Rate equality)"""
        try:
            group_rates = {}
            for group in groups:
                group_mask = sensitive_attr.astype(str) == group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                # Calculate True Positive Rate
                if group_mask.sum() > 0 and (group_y_true == 1).sum() > 0:
                    tpr = ((group_y_true == 1) & (group_y_pred == 1)).sum() / (group_y_true == 1).sum()
                    group_rates[group] = round(tpr, 3)
                else:
                    group_rates[group] = 0.0
            
            # Calculate maximum difference in TPR
            if len(group_rates) > 1:
                rates = list(group_rates.values())
                max_diff = max(rates) - min(rates)
            else:
                max_diff = 0.0
            
            status = MetricStatus.FAIR if max_diff <= self.thresholds['equal_opportunity'] else MetricStatus.BIASED
            
            return FairnessMetric(
                name="Equal Opportunity",
                value=round(max_diff, 3),
                status=status,
                threshold=self.thresholds['equal_opportunity'],
                groupRates=group_rates,
                description="Difference in True Positive Rates across groups",
                tooltip="Lower values indicate better fairness. Values > 0.10 suggest bias in positive outcomes."
            )
            
        except Exception as e:
            logger.warning(f"Equal opportunity calculation failed: {str(e)}")
            return self._create_error_metric("Equal Opportunity", str(e))
    
    def _calculate_equalized_odds(self, sensitive_attr: pd.Series, y_true: pd.Series, y_pred: pd.Series, groups: List[str]) -> FairnessMetric:
        """Calculate Equalized Odds (TPR and FPR equality)"""
        try:
            group_rates = {}
            tpr_rates = []
            fpr_rates = []
            
            for group in groups:
                group_mask = sensitive_attr.astype(str) == group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                if group_mask.sum() > 0:
                    # Calculate TPR and FPR
                    if (group_y_true == 1).sum() > 0:
                        tpr = ((group_y_true == 1) & (group_y_pred == 1)).sum() / (group_y_true == 1).sum()
                    else:
                        tpr = 0.0
                    
                    if (group_y_true == 0).sum() > 0:
                        fpr = ((group_y_true == 0) & (group_y_pred == 1)).sum() / (group_y_true == 0).sum()
                    else:
                        fpr = 0.0
                    
                    tpr_rates.append(tpr)
                    fpr_rates.append(fpr)
                    group_rates[group] = round((tpr + (1 - fpr)) / 2, 3)  # Combined metric
                else:
                    group_rates[group] = 0.0
            
            # Calculate maximum difference in equalized odds
            if len(tpr_rates) > 1 and len(fpr_rates) > 1:
                tpr_diff = max(tpr_rates) - min(tpr_rates)
                fpr_diff = max(fpr_rates) - min(fpr_rates)
                max_diff = max(tpr_diff, fpr_diff)
            else:
                max_diff = 0.0
            
            status = MetricStatus.FAIR if max_diff <= self.thresholds['equalized_odds'] else MetricStatus.BIASED
            
            return FairnessMetric(
                name="Equalized Odds",
                value=round(max_diff, 3),
                status=status,
                threshold=self.thresholds['equalized_odds'],
                groupRates=group_rates,
                description="Maximum difference in TPR and FPR across groups",
                tooltip="Lower values indicate better fairness. Values > 0.10 suggest bias in prediction accuracy."
            )
            
        except Exception as e:
            logger.warning(f"Equalized odds calculation failed: {str(e)}")
            return self._create_error_metric("Equalized Odds", str(e))
    
    def _calculate_calibration(self, sensitive_attr: pd.Series, y_true: pd.Series, y_pred_proba: np.ndarray, groups: List[str]) -> FairnessMetric:
        """Calculate Calibration (predicted probability vs actual outcome)"""
        try:
            group_rates = {}
            
            # For binary classification, use positive class probabilities
            if y_pred_proba.ndim > 1:
                proba_positive = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
            else:
                proba_positive = y_pred_proba
            
            calibration_diffs = []
            
            for group in groups:
                group_mask = sensitive_attr.astype(str) == group
                group_y_true = y_true[group_mask]
                group_proba = proba_positive[group_mask]
                
                if group_mask.sum() > 0:
                    # Calculate calibration in bins
                    n_bins = 5
                    bin_edges = np.linspace(0, 1, n_bins + 1)
                    bin_diffs = []
                    
                    for i in range(n_bins):
                        bin_mask = (group_proba >= bin_edges[i]) & (group_proba < bin_edges[i + 1])
                        if i == n_bins - 1:  # Include upper bound for last bin
                            bin_mask = (group_proba >= bin_edges[i]) & (group_proba <= bin_edges[i + 1])
                        
                        if bin_mask.sum() > 0:
                            mean_pred_proba = group_proba[bin_mask].mean()
                            actual_positive_rate = group_y_true[bin_mask].mean()
                            bin_diff = abs(mean_pred_proba - actual_positive_rate)
                            bin_diffs.append(bin_diff)
                    
                    if bin_diffs:
                        group_calibration = np.mean(bin_diffs)
                        calibration_diffs.append(group_calibration)
                        group_rates[group] = round(group_calibration, 3)
                    else:
                        group_rates[group] = 0.0
                else:
                    group_rates[group] = 0.0
            
            # Calculate maximum calibration difference
            max_diff = max(calibration_diffs) if calibration_diffs else 0.0
            
            status = MetricStatus.FAIR if max_diff <= self.thresholds['calibration'] else MetricStatus.BIASED
            
            return FairnessMetric(
                name="Calibration",
                value=round(max_diff, 3),
                status=status,
                threshold=self.thresholds['calibration'],
                groupRates=group_rates,
                description="Difference in calibration (predicted vs actual probabilities) across groups",
                tooltip="Lower values indicate better calibration fairness. Values > 0.10 suggest miscalibration."
            )
            
        except Exception as e:
            logger.warning(f"Calibration calculation failed: {str(e)}")
            return self._create_error_metric("Calibration", str(e))
    
    def _calculate_generalized_entropy_index(self, sensitive_attr: pd.Series, y_pred: pd.Series, groups: List[str]) -> FairnessMetric:
        """Calculate Generalized Entropy Index"""
        try:
            group_rates = {}
            
            # Calculate benefit (positive prediction rate) for each individual
            overall_benefit = y_pred.mean()
            
            if overall_benefit == 0:
                return FairnessMetric(
                    name="Generalized Entropy Index",
                    value=0.0,
                    status=MetricStatus.FAIR,
                    threshold=self.thresholds['generalized_entropy_index'],
                    groupRates={group: 0.0 for group in groups},
                    description="Entropy-based fairness measure",
                    tooltip="No positive predictions to analyze"
                )
            
            entropy_components = []
            
            for group in groups:
                group_mask = sensitive_attr.astype(str) == group
                if group_mask.sum() > 0:
                    group_benefit = y_pred[group_mask].mean()
                    group_size = group_mask.sum()
                    total_size = len(y_pred)
                    
                    if group_benefit > 0:
                        # Calculate entropy component for this group
                        ratio = group_benefit / overall_benefit
                        entropy_component = (group_size / total_size) * ratio * np.log(ratio)
                        entropy_components.append(entropy_component)
                        group_rates[group] = round(group_benefit, 3)
                    else:
                        group_rates[group] = 0.0
                else:
                    group_rates[group] = 0.0
            
            # Calculate generalized entropy index
            gei = sum(entropy_components) if entropy_components else 0.0
            
            status = MetricStatus.FAIR if abs(gei) <= self.thresholds['generalized_entropy_index'] else MetricStatus.BIASED
            
            return FairnessMetric(
                name="Generalized Entropy Index",
                value=round(abs(gei), 3),
                status=status,
                threshold=self.thresholds['generalized_entropy_index'],
                groupRates=group_rates,
                description="Entropy-based measure of fairness across groups",
                tooltip="Lower values indicate better fairness. Values > 0.30 suggest significant inequality."
            )
            
        except Exception as e:
            logger.warning(f"Generalized entropy index calculation failed: {str(e)}")
            return self._create_error_metric("Generalized Entropy Index", str(e))
    
    def _create_error_metric(self, name: str, error: str) -> FairnessMetric:
        """Create error metric for failed calculations"""
        return FairnessMetric(
            name=name,
            value=0.0,
            status=MetricStatus.WARNING,
            threshold=0.0,
            groupRates={},
            description=f"Calculation failed: {error}",
            tooltip="This metric could not be calculated due to data issues"
        )
    
    def _calculate_overall_score(self, metrics: List[FairnessMetric]) -> float:
        """Calculate overall fairness score (0-100)"""
        try:
            scores = []
            
            for metric in metrics:
                if metric.status == MetricStatus.FAIR:
                    scores.append(100)
                elif metric.status == MetricStatus.WARNING:
                    scores.append(50)
                else:  # BIASED
                    # Calculate score based on how far from threshold
                    if metric.threshold > 0:
                        excess = max(0, metric.value - metric.threshold)
                        penalty = min(100, (excess / metric.threshold) * 50)
                        scores.append(max(0, 100 - penalty))
                    else:
                        scores.append(0)
            
            return round(np.mean(scores) if scores else 0, 1)
            
        except Exception:
            return 0.0
    
    def _determine_overall_rating(self, score: float) -> OverallRating:
        """Determine overall fairness rating"""
        if score >= 80:
            return OverallRating.GOOD_FAIRNESS
        elif score >= 60:
            return OverallRating.MODERATE_FAIRNESS
        else:
            return OverallRating.POOR_FAIRNESS
    
    def _determine_risk_classification(self, metrics: List[FairnessMetric]) -> str:
        """Determine risk classification"""
        biased_count = sum(1 for m in metrics if m.status == MetricStatus.BIASED)
        warning_count = sum(1 for m in metrics if m.status == MetricStatus.WARNING)
        
        if biased_count >= 3:
            return "High Bias Risk"
        elif biased_count >= 2 or (biased_count >= 1 and warning_count >= 2):
            return "Medium Bias Risk"
        else:
            return "Low Bias Risk"
    
    def _identify_primary_issues(self, metrics: List[FairnessMetric], feature_name: str) -> List[str]:
        """Identify primary bias issues"""
        issues = []
        
        for metric in metrics:
            if metric.status == MetricStatus.BIASED:
                if "Statistical Parity" in metric.name:
                    issues.append(f"{feature_name} selection rate discrimination")
                elif "Disparate Impact" in metric.name:
                    issues.append(f"{feature_name} disparate impact")
                elif "Equal Opportunity" in metric.name:
                    issues.append(f"{feature_name} unequal opportunity")
                elif "Equalized Odds" in metric.name:
                    issues.append(f"{feature_name} prediction accuracy bias")
                elif "Calibration" in metric.name:
                    issues.append(f"{feature_name} calibration bias")
                elif "Entropy" in metric.name:
                    issues.append(f"{feature_name} distributional unfairness")
        
        return issues[:3]  # Limit to top 3 issues
    
    def _create_default_metrics(self, feature_name: str, groups: List[str]) -> AttributeFairnessMetrics:
        """Create default metrics for insufficient groups"""
        default_metric = FairnessMetric(
            name="Insufficient Data",
            value=0.0,
            status=MetricStatus.WARNING,
            threshold=0.0,
            groupRates={group: 0.0 for group in groups},
            description="Insufficient groups for fairness analysis",
            tooltip="This feature has insufficient group diversity for meaningful fairness analysis"
        )
        
        return AttributeFairnessMetrics(
            attribute=feature_name,
            groups=groups,
            metrics=[default_metric],
            overallScore=50.0,
            overallRating=OverallRating.MODERATE_FAIRNESS,
            riskClassification="Unable to Assess",
            primaryIssues=["Insufficient group diversity"]
        )
