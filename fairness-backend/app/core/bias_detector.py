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
    
    def __init__(self, significance_level: float = 0.05, use_hsic: bool = True, kernel_type: str = 'rbf'):
        self.significance_level = significance_level
        self.sensitive_features = []
        self.statistical_results = {}
        self.use_hsic = use_hsic
        self.kernel_type = kernel_type
        self.epsilon = 1e-6  # Regularization parameter
        
    def detect_sensitive_features(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        feature_types: Dict[str, str],
        exclude_columns: Optional[List[str]] = None
    ) -> SensitiveFeatureDetectionResponse:
        """
        Main method to detect sensitive features using HSIC/NOCCO approach and statistical tests
        """
        exclude_columns = exclude_columns or []
        
        try:
            if self.use_hsic:
                # Perform HSIC/NOCCO analysis for sensitive feature detection
                statistical_results = self._perform_hsic_analysis(
                    df, target_column, feature_types, exclude_columns
                )
            else:
                # Fallback to traditional statistical tests
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
            logger.error(f"Bias detection failed: {str(e)}", exc_info=True)
            raise
    
    def _perform_hsic_analysis(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        feature_types: Dict[str, str],
        exclude_columns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform HSIC/NOCCO analysis to detect sensitive features based on the research paper
        """
        results = {}
        target_values = df[target_column].values

        # Remove rows with missing target values
        valid_target_mask = ~pd.isna(target_values)
        
        # Calculate NOCCO values for all features
        nocco_values = {}
        
        for feature, feature_type in feature_types.items():
            if feature == target_column or feature in exclude_columns:
                continue
                
            try:
                # Get valid indices for this feature
                feature_series = df[feature]
                valid_mask = valid_target_mask & ~feature_series.isnull()
                
                if valid_mask.sum() < 10:
                    results[feature] = {
                        'test_type': 'hsic_insufficient_data',
                        'p_value': 1.0,
                        'statistic': None,
                        'correlation': 0.0,
                        'is_sensitive': False
                    }
                    continue

                # Handle feature based on its type
                if feature_type == 'categorical':
                    # For categorical features, perform one-hot encoding and find max NOCCO
                    feature_result = self._analyze_categorical_feature(feature_series[valid_mask], target_values[valid_mask])
                else:
                    # For numerical features, calculate NOCCO directly
                    feature_result = self._analyze_numerical_feature(feature_series[valid_mask], target_values[valid_mask])
                
                results[feature] = feature_result
                nocco_values[feature] = feature_result['correlation']  # Using correlation for NOCCO value
                        
            except Exception as e:
                logger.warning(f"HSIC analysis failed for feature {feature}: {str(e)}")
                results[feature] = {
                    'test_type': 'hsic_failed',
                    'p_value': 1.0,
                    'statistic': None,
                    'correlation': 0.0,
                    'error': str(e),
                    'is_sensitive': False
                }

        # Calculate threshold (median of NOCCO values)
        if nocco_values:
            threshold = np.percentile(list(nocco_values.values()), 75)
            logger.info(f"NOCCO threshold (median): {threshold:.4f}")
        else:
            threshold = 0.0
            
        # Mark features as sensitive based on threshold
        for feature, result in results.items():
            if 'is_sensitive' not in result:
                if result['correlation'] >= threshold:
                    if feature not in self.sensitive_features:
                        self.sensitive_features.append(feature)
                    result['is_sensitive'] = True
                else:
                    result['is_sensitive'] = False
        
        self.statistical_results = results
        return results
    
    def _analyze_categorical_feature(self, feature_series: pd.Series, target_values: np.ndarray) -> Dict[str, Any]:
        """
        Analyze categorical feature using NOCCO approach (one-vs-all for each category)
        """
        # # Remove missing values
        # valid_mask = ~feature_series.isnull()
        # feature_clean = feature_series[valid_mask]
        # target_clean = target_values[valid_mask]
        
        # Get unique categories
        categories = feature_series.unique()

        if len(categories) < 2:
            return {
                'test_type': 'hsic_insufficient_groups',
                'p_value': 1.0,
                'statistic': None,
                'correlation': 0.0
            }
        
        # Calculate NOCCO for each category (one-vs-all)
        max_nocco = 0.0
        max_category = None
        category_noccos = {}
        
        for category in categories:
            # Create binary indicator for this category
            category_indicator = (feature_series == category).astype(float).values.reshape(-1, 1)
            
            # Calculate NOCCO
            nocco_value = self._calculate_nocco(category_indicator, target_values.reshape(-1, 1))

            # Store NOCCO value for each category
            category_noccos[str(category)] = nocco_value
            
            if nocco_value > max_nocco:
                max_nocco = nocco_value
                max_category = category
        
        # Estimate p-value based on NOCCO value (higher NOCCO = lower p-value)
        # This is a heuristic approximation as the paper doesn't specify how to convert NOCCO to p-value
        p_value = np.exp(-5 * max_nocco)  # Exponential decay function
        p_value = np.clip(p_value, 0.0, 1.0)  # Ensure p-value is in [0,1]
        
        return {
            'test_type': 'hsic_nocco',
            'p_value': p_value,
            'statistic': max_nocco,
            'correlation': max_nocco,
            'sensitive_group': str(max_category),
            'category_noccos': category_noccos
        }
    
    def _analyze_numerical_feature(self, feature_series: pd.Series, target_values: np.ndarray) -> Dict[str, Any]:
        """
        Analyze numerical feature using NOCCO approach
        """
        # Remove missing values
        feature_clean = feature_series.values.reshape(-1, 1)
        target_clean = target_values.reshape(-1, 1)
        
        # Calculate NOCCO
        nocco_value = self._calculate_nocco(feature_clean, target_clean)
        
        # Estimate p-value based on NOCCO value
        p_value = np.exp(-5 * nocco_value)  # Exponential decay function
        p_value = np.clip(p_value, 0.0, 1.0)  # Ensure p-value is in [0,1]
        
        return {
            'test_type': 'hsic_nocco',
            'p_value': p_value,
            'statistic': nocco_value,
            'correlation': nocco_value
        }
    
    def _calculate_nocco(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate NOrmalized Cross-Covariance Operator (NOCCO)
        Based on the paper's algorithm
        """
        n = X.shape[0]
        
        try:
            # Calculate kernel matrices
            K_x = self._kernel_matrix(X)
            K_y = self._kernel_matrix(y)
            
            # Centering matrix
            H = np.eye(n) - (1.0/n) * np.ones((n, n))
            
            # Centered kernel matrices
            K_x_centered = H @ K_x @ H
            K_y_centered = H @ K_y @ H
    
            # Calculate R_x and R_y with regularization
            # R_x = K_x_centered @ inv(K_x_centered + epsilon * n * I)
            K_x_reg = K_x_centered + self.epsilon * n * np.eye(n)
            K_y_reg = K_y_centered + self.epsilon * n * np.eye(n)
            
            R_x = K_x_centered @ np.linalg.inv(K_x_reg)
            R_y = K_y_centered @ np.linalg.inv(K_y_reg)
            
            # Calculate NOCCO
            nocco_value = np.trace(R_x @ R_y) / (np.sqrt(np.trace(R_x @ R_x)) * np.sqrt(np.trace(R_y @ R_y)) + self.epsilon)
            
            # Ensure value is between 0 and 1
            nocco_value = np.clip(nocco_value, 0.0, 1.0)
            
            return float(nocco_value)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Matrix inversion failed in NOCCO calculation: {str(e)}")
            return 0.0
    
    def _kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate kernel matrix for a given vector or matrix X
        """
        n = X.shape[0]
        
        if self.kernel_type == 'rbf':
            # Calculate pairwise squared Euclidean distances
            X_squared = np.sum(X**2, axis=1).reshape(-1, 1)
            distances = X_squared + X_squared.T - 2 * np.dot(X, X.T)

            # Ensure non-negative distances (numerical stability)
            distances = np.maximum(distances, 0)

            # Apply RBF kernel with sigma = 1/n
            sigma = np.median(distances[distances > 0])
            K = np.exp(-distances / (2 * sigma**2 + self.epsilon))
        elif self.kernel_type == 'linear':
            # Linear kernel is just the dot product
            K = np.dot(X, X.T)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return K
    
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
                    test_result['is_sensitive'] = True
                else:
                    test_result['is_sensitive'] = False

            except Exception as e:
                logger.warning(f"Statistical test failed for feature {feature}: {str(e)}")
                results[feature] = {
                    'test_type': 'failed',
                    'p_value': 1.0,
                    'statistic': None,
                    'correlation': 0.0,
                    'error': str(e),
                    'is_sensitive': False
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
                'p_value': float(p_value),
                'statistic': float(correlation),
                'correlation': float(abs(correlation))
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
                'p_value': float(p_value),
                'statistic': float(f_statistic),
                'correlation': float(correlation)
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
                'p_value': float(p_value),
                'statistic': float(chi2_stat),
                'correlation': float(correlation)
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
                # Skip failed tests or insufficient data
                if result.get('test_type') in ['failed', 'hsic_insufficient_data', 'hsic_failed']:
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
                is_sensitive = result.get('is_sensitive', False)
                sensitivity_level = self._classify_sensitivity(p_value, correlation, is_sensitive)
                
                # Get unique groups
                groups = self._get_feature_groups(df[feature_name])

                # Add sensitive group information if available
                sensitive_group = result.get('sensitive_group')
                if sensitive_group is not None and str(sensitive_group) in groups:
                    # Move sensitive group to front
                    groups = [str(sensitive_group)] + [g for g in groups if g != str(sensitive_group)]
                
                # Generate description
                description = self._generate_feature_description(
                    feature_name, test_type, p_value, correlation, groups,
                    is_hsic=result.get('test_type', '').startswith('hsic')
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
        
        # Sort by correlation (highest first) for HSIC, or p-value for traditional tests
        if self.use_hsic:
            features.sort(key=lambda x: x.correlation, reverse=True)
        else:
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
            't_test': TestType.T_TEST,
            'hsic_nocco': TestType.HSIC if hasattr(TestType, 'HSIC') else TestType.CHI_SQUARE
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
    
    def _classify_sensitivity(self, p_value: float, correlation: float, is_sensitive: bool = False) -> SensitivityLevel:
        """Classify sensitivity level based on p-value, effect size, and HSIC threshold"""
        abs_corr = abs(correlation)
        
        if self.use_hsic:
            # For HSIC, use the is_sensitive flag and correlation strength
            if is_sensitive:
                if abs_corr >= 0.5:
                    return SensitivityLevel.HIGHLY_SENSITIVE
                elif abs_corr >= 0.3:
                    return SensitivityLevel.MODERATELY_SENSITIVE
                else:
                    return SensitivityLevel.LOW_SENSITIVITY
            else:
                return SensitivityLevel.LOW_SENSITIVITY
        else:
            # Traditional approach based on p-value
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
            return [str(val) for val in sorted(unique_values, key=str)]
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
        groups: List[str],
        is_hsic: bool = False
    ) -> str:
        """Generate human-readable description"""
        effect_magnitude = "strong" if abs(correlation) >= 0.5 else "moderate" if abs(correlation) >= 0.3 else "weak"
        
        group_info = ""
        if len(groups) <= 3:
            group_info = f" across groups: {', '.join(groups)}"
        elif len(groups) > 0:
            group_info = f" across {len(groups)} groups"
        
        if is_hsic:
            significance = "high" if correlation >= 0.5 else "moderate" if correlation >= 0.3 else "low"
            return f"Feature '{feature_name}' shows {significance} dependence with target (HSIC/NOCCO={correlation:.3f}) with {effect_magnitude} effect size{group_info}."
        else:
            significance = "highly significant" if p_value < 0.01 else "significant" if p_value < 0.05 else "not significant"
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
