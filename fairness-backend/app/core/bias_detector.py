"""
Bias detection module adapted from backend_old with FastAPI integration
Performs statistical tests to identify sensitive features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
import logging

from app.models.schemas import (
    SensitiveFeature, SensitiveFeatureSummary, SensitiveFeatureDetectionResponse,
    DataType, TestType, EffectSizeLabel, SensitivityLevel, RiskLevel
)

logger = logging.getLogger(__name__)


class BiasDetector:
    """Enhanced bias detection with comprehensive statistical analysis"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.sensitive_features = []
        self.statistical_results = {}
        
    def detect_sensitive_features(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        feature_types: Dict[str, str],
        exclude_columns: Optional[List[str]] = None
    ) -> SensitiveFeatureDetectionResponse:
        """
        Main method to detect sensitive features using statistical tests
        """
        exclude_columns = exclude_columns or []
        
        try:
            # Perform statistical tests
            statistical_results = self._perform_statistical_tests(
                df, target_column, feature_types, exclude_columns
            )
            
            # Convert results to Pydantic models
            detected_features = self._convert_to_response_format(statistical_results, df)
            
            # Generate summary
            summary = self._generate_summary(detected_features)
            
            logger.info(f"Bias detection completed. Found {len(detected_features)} sensitive features")
            
            return SensitiveFeatureDetectionResponse(
                detectedFeatures=detected_features,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Bias detection failed: {str(e)}")
            raise
    
    def _perform_statistical_tests(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        feature_types: Dict[str, str],
        exclude_columns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform appropriate statistical tests based on feature and target types
        """
        results = {}
        target_type = self._determine_target_type(df[target_column])
        
        for feature, feature_type in feature_types.items():
            if feature == target_column or feature in exclude_columns:
                continue
                
            try:
                test_result = self._perform_feature_test(
                    df[feature], df[target_column], feature_type, target_type
                )
                results[feature] = test_result
                
                # Mark as sensitive if p-value is below significance level
                if test_result['p_value'] < self.significance_level:
                    if feature not in self.sensitive_features:
                        self.sensitive_features.append(feature)
                        
            except Exception as e:
                logger.warning(f"Statistical test failed for feature {feature}: {str(e)}")
                results[feature] = {
                    'test_type': 'failed',
                    'p_value': 1.0,
                    'statistic': None,
                    'correlation': 0.0,
                    'error': str(e)
                }
        
        self.statistical_results = results
        return results
    
    def _determine_target_type(self, target_series: pd.Series) -> str:
        """Determine if target is categorical or numerical"""
        if target_series.dtype in ['object', 'category', 'bool']:
            return 'categorical'
        else:
            unique_values = target_series.nunique()
            if unique_values <= 10:
                return 'categorical'
            else:
                return 'numerical'
    
    def _perform_feature_test(
        self, 
        feature_series: pd.Series, 
        target_series: pd.Series,
        feature_type: str, 
        target_type: str
    ) -> Dict[str, Any]:
        """
        Perform appropriate statistical test based on feature and target types
        """
        # Remove missing values
        valid_mask = ~(feature_series.isnull() | target_series.isnull())
        feature_clean = feature_series[valid_mask]
        target_clean = target_series[valid_mask]
        
        if len(feature_clean) < 10:
            return {
                'test_type': 'insufficient_data',
                'p_value': 1.0,
                'statistic': None,
                'correlation': 0.0
            }
        
        # Numerical feature, numerical target - Pearson correlation
        if feature_type == 'numerical' and target_type == 'numerical':
            return self._pearson_correlation_test(feature_clean, target_clean)
        
        # Categorical feature, numerical target - ANOVA F-test
        elif feature_type == 'categorical' and target_type == 'numerical':
            return self._anova_test(feature_clean, target_clean)
        
        # Categorical feature, categorical target - Chi-square test
        elif feature_type == 'categorical' and target_type == 'categorical':
            return self._chi_square_test(feature_clean, target_clean)
        
        # Numerical feature, categorical target - ANOVA (reversed)
        elif feature_type == 'numerical' and target_type == 'categorical':
            return self._anova_test(target_clean, feature_clean, reversed=True)
        
        else:
            return {
                'test_type': 'unsupported',
                'p_value': 1.0,
                'statistic': None,
                'correlation': 0.0
            }
    
    def _pearson_correlation_test(self, feature: pd.Series, target: pd.Series) -> Dict[str, Any]:
        """Perform Pearson correlation test"""
        try:
            correlation, p_value = stats.pearsonr(feature, target)
            return {
                'test_type': 'pearson_correlation',
                'p_value': p_value,
                'statistic': correlation,
                'correlation': abs(correlation)
            }
        except Exception as e:
            logger.warning(f"Pearson correlation test failed: {str(e)}")
            return {
                'test_type': 'failed',
                'p_value': 1.0,
                'statistic': None,
                'correlation': 0.0,
                'error': str(e)
            }
    
    def _anova_test(self, groups: pd.Series, values: pd.Series, reversed: bool = False) -> Dict[str, Any]:
        """Perform ANOVA F-test"""
        try:
            # Group values by categories
            grouped_data = []
            unique_groups = groups.dropna().unique()
            
            if len(unique_groups) < 2:
                return {
                    'test_type': 'insufficient_groups',
                    'p_value': 1.0,
                    'statistic': None,
                    'correlation': 0.0
                }
            
            for group in unique_groups:
                group_values = values[groups == group].dropna()
                if len(group_values) > 0:
                    grouped_data.append(group_values)
            
            if len(grouped_data) < 2:
                return {
                    'test_type': 'insufficient_groups',
                    'p_value': 1.0,
                    'statistic': None,
                    'correlation': 0.0
                }
            
            # Perform ANOVA
            f_statistic, p_value = f_oneway(*grouped_data)
            
            # Calculate effect size (eta-squared)
            correlation = self._calculate_eta_squared(grouped_data, f_statistic)
            
            return {
                'test_type': 'anova_f_test',
                'p_value': p_value,
                'statistic': f_statistic,
                'correlation': correlation
            }
            
        except Exception as e:
            logger.warning(f"ANOVA test failed: {str(e)}")
            return {
                'test_type': 'failed',
                'p_value': 1.0,
                'statistic': None,
                'correlation': 0.0,
                'error': str(e)
            }
    
    def _chi_square_test(self, feature: pd.Series, target: pd.Series) -> Dict[str, Any]:
        """Perform Chi-square test of independence"""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(feature, target)
            
            # Check if contingency table is valid
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return {
                    'test_type': 'insufficient_groups',
                    'p_value': 1.0,
                    'statistic': None,
                    'correlation': 0.0
                }
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate Cramer's V (effect size)
            n = contingency_table.sum().sum()
            correlation = self._calculate_cramers_v(chi2_stat, n, contingency_table.shape)
            
            return {
                'test_type': 'chi_square',
                'p_value': p_value,
                'statistic': chi2_stat,
                'correlation': correlation
            }
            
        except Exception as e:
            logger.warning(f"Chi-square test failed: {str(e)}")
            return {
                'test_type': 'failed',
                'p_value': 1.0,
                'statistic': None,
                'correlation': 0.0,
                'error': str(e)
            }
    
    def _calculate_eta_squared(self, grouped_data: List[pd.Series], f_statistic: float) -> float:
        """Calculate eta-squared (effect size for ANOVA)"""
        try:
            # Calculate between-group and within-group sum of squares
            all_data = pd.concat(grouped_data)
            grand_mean = all_data.mean()
            
            # Between-group sum of squares
            ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in grouped_data)
            
            # Total sum of squares
            ss_total = sum((all_data - grand_mean) ** 2)
            
            if ss_total == 0:
                return 0.0
            
            eta_squared = ss_between / ss_total
            return min(eta_squared, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0
    
    def _calculate_cramers_v(self, chi2_stat: float, n: int, shape: Tuple[int, int]) -> float:
        """Calculate Cramer's V (effect size for chi-square)"""
        try:
            if n == 0:
                return 0.0
            
            min_dim = min(shape[0] - 1, shape[1] - 1)
            if min_dim == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2_stat / (n * min_dim))
            return min(cramers_v, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0
    
    def _convert_to_response_format(
        self, 
        statistical_results: Dict[str, Dict[str, Any]], 
        df: pd.DataFrame
    ) -> List[SensitiveFeature]:
        """
        Convert statistical results to Pydantic response format
        """
        features = []
        
        for feature_name, result in statistical_results.items():
            try:
                # Skip failed tests
                if result.get('test_type') == 'failed':
                    continue
                
                # Determine data type
                data_type = self._get_data_type(df[feature_name])
                
                # Determine test type
                test_type = self._get_test_type(result.get('test_type', ''))
                
                # Calculate effect size and classification
                correlation = result.get('correlation', 0.0)
                effect_size_label = self._classify_effect_size(correlation)
                
                # Determine sensitivity level
                p_value = result.get('p_value', 1.0)
                sensitivity_level = self._classify_sensitivity(p_value, correlation)
                
                # Get unique groups
                groups = self._get_feature_groups(df[feature_name])
                
                # Generate description
                description = self._generate_feature_description(
                    feature_name, test_type, p_value, correlation, groups
                )
                
                feature = SensitiveFeature(
                    name=feature_name,
                    dataType=data_type,
                    test=test_type,
                    pValue=round(p_value, 6),
                    effectSize=round(correlation, 3),
                    effectSizeLabel=effect_size_label,
                    correlation=round(correlation, 3),
                    sensitivityLevel=sensitivity_level,
                    groups=groups,
                    description=description
                )
                
                features.append(feature)
                
            except Exception as e:
                logger.warning(f"Failed to convert feature {feature_name}: {str(e)}")
                continue
        
        # Sort by p-value (most significant first)
        features.sort(key=lambda x: x.pValue)
        
        return features
    
    def _get_data_type(self, series: pd.Series) -> DataType:
        """Determine Pydantic DataType enum"""
        unique_values = series.nunique()
        
        if series.dtype in ['object', 'category']:
            return DataType.CATEGORICAL
        elif unique_values == 2:
            return DataType.BINARY
        elif unique_values <= 10:
            return DataType.CATEGORICAL
        else:
            return DataType.NUMERICAL
    
    def _get_test_type(self, test_type_str: str) -> TestType:
        """Convert test type string to Pydantic enum"""
        mapping = {
            'chi_square': TestType.CHI_SQUARE,
            'anova_f_test': TestType.ANOVA,
            'pearson_correlation': TestType.PEARSON,
            't_test': TestType.T_TEST
        }
        return mapping.get(test_type_str, TestType.CHI_SQUARE)
    
    def _classify_effect_size(self, correlation: float) -> EffectSizeLabel:
        """Classify effect size based on correlation/effect size"""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.5:
            return EffectSizeLabel.LARGE
        elif abs_corr >= 0.3:
            return EffectSizeLabel.MEDIUM
        else:
            return EffectSizeLabel.SMALL
    
    def _classify_sensitivity(self, p_value: float, correlation: float) -> SensitivityLevel:
        """Classify sensitivity level based on p-value and effect size"""
        abs_corr = abs(correlation)
        
        if p_value < 0.01 and abs_corr >= 0.3:
            return SensitivityLevel.HIGHLY_SENSITIVE
        elif p_value < 0.05 and abs_corr >= 0.1:
            return SensitivityLevel.MODERATELY_SENSITIVE
        else:
            return SensitivityLevel.LOW_SENSITIVITY
    
    def _get_feature_groups(self, series: pd.Series) -> List[str]:
        """Get unique groups/values in feature"""
        unique_values = series.dropna().unique()
        
        # Limit to reasonable number of groups for display
        if len(unique_values) <= 10:
            return [str(val) for val in sorted(unique_values)]
        else:
            # For high cardinality, show top 5 most frequent
            top_values = series.value_counts().head(5).index
            return [str(val) for val in top_values] + [f"... and {len(unique_values) - 5} more"]
    
    def _generate_feature_description(
        self, 
        feature_name: str, 
        test_type: TestType, 
        p_value: float, 
        correlation: float,
        groups: List[str]
    ) -> str:
        """Generate human-readable description"""
        significance = "highly significant" if p_value < 0.01 else "significant" if p_value < 0.05 else "not significant"
        effect_magnitude = "strong" if abs(correlation) >= 0.5 else "moderate" if abs(correlation) >= 0.3 else "weak"
        
        group_info = f" across groups: {', '.join(groups[:3])}" if len(groups) <= 3 else f" across {len(groups)} groups"
        
        return f"Feature '{feature_name}' shows {significance} association with target ({test_type.value}, p={p_value:.4f}) with {effect_magnitude} effect size{group_info}."
    
    def _generate_summary(self, features: List[SensitiveFeature]) -> SensitiveFeatureSummary:
        """Generate summary of detection results"""
        total_detected = len(features)
        highly_sensitive = sum(1 for f in features if f.sensitivityLevel == SensitivityLevel.HIGHLY_SENSITIVE)
        moderately_sensitive = sum(1 for f in features if f.sensitivityLevel == SensitivityLevel.MODERATELY_SENSITIVE)
        
        # Determine overall risk level
        if highly_sensitive >= 3 or (highly_sensitive >= 1 and moderately_sensitive >= 2):
            risk_level = RiskLevel.HIGH
        elif highly_sensitive >= 1 or moderately_sensitive >= 2:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return SensitiveFeatureSummary(
            totalDetected=total_detected,
            highlySensitiveCount=highly_sensitive,
            moderatelySensitiveCount=moderately_sensitive,
            riskLevel=risk_level
        )
