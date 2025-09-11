"""
Data validation utilities
Validates datasets and models for fairness analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from app.models.schemas import ValidationResult

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data for fairness analysis"""
    
    def __init__(self):
        self.min_rows = 100  # Minimum rows for meaningful analysis
        self.min_groups = 2  # Minimum groups for bias analysis
        self.max_missing_percent = 0.5  # Maximum 50% missing values
    
    def validate_dataset(self, df: pd.DataFrame, is_training: bool = True) -> ValidationResult:
        """Validate dataset for fairness analysis"""
        errors = []
        warnings = []
        
        try:
            # Basic validations
            if df.empty:
                errors.append("Dataset is empty")
                return ValidationResult(isValid=False, errors=errors, warnings=warnings)
            
            # Check minimum rows
            if len(df) < self.min_rows:
                warnings.append(f"Dataset has only {len(df)} rows. Recommended minimum is {self.min_rows} for meaningful analysis")
            
            # Check for missing values
            missing_percent = df.isnull().sum() / len(df)
            high_missing_cols = missing_percent[missing_percent > self.max_missing_percent].index.tolist()
            
            if high_missing_cols:
                warnings.append(f"Columns with high missing values (>{self.max_missing_percent*100}%): {', '.join(high_missing_cols)}")
            
            # Check data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_cols) == 0 and len(categorical_cols) == 0:
                errors.append("No valid columns found for analysis")
            
            # Check for constant columns
            constant_cols = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                warnings.append(f"Columns with constant values (may not be useful): {', '.join(constant_cols)}")
            
            # Training-specific validations
            if is_training:
                # Check for potential target columns (binary or categorical with few unique values)
                potential_targets = []
                for col in df.columns:
                    unique_count = df[col].nunique()
                    if 2 <= unique_count <= 10:
                        potential_targets.append(f"{col} ({unique_count} values)")
                
                if potential_targets:
                    warnings.append(f"Potential target columns detected: {', '.join(potential_targets)}")
            
            # Check for potential sensitive attributes
            potential_sensitive = []
            sensitive_keywords = ['gender', 'race', 'age', 'ethnic', 'religion', 'sex', 'nationality']
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in sensitive_keywords):
                    potential_sensitive.append(col)
            
            if potential_sensitive:
                warnings.append(f"Potential sensitive attributes detected: {', '.join(potential_sensitive)}")
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                warnings.append(f"Found {duplicate_count} duplicate rows")
            
            # Data quality checks
            for col in numeric_cols:
                # Check for outliers (very basic check)
                if df[col].std() > 0:  # Avoid division by zero
                    outliers = df[(np.abs(df[col] - df[col].mean()) > 3 * df[col].std())].shape[0]
                    if outliers > len(df) * 0.05:  # More than 5% outliers
                        warnings.append(f"Column '{col}' may have outliers ({outliers} rows)")
            
            # Check for high cardinality categorical columns
            for col in categorical_cols:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5:  # More than 50% unique values
                    warnings.append(f"Column '{col}' has high cardinality ({df[col].nunique()} unique values)")
            
            is_valid = len(errors) == 0
            
            logger.info(f"Dataset validation completed: {len(df)} rows, {len(df.columns)} columns, "
                       f"{len(errors)} errors, {len(warnings)} warnings")
            
            return ValidationResult(isValid=is_valid, errors=errors, warnings=warnings)
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return ValidationResult(
                isValid=False, 
                errors=[f"Validation error: {str(e)}"], 
                warnings=warnings
            )
    
    def validate_model(self, model: Any) -> ValidationResult:
        """Validate ML model for fairness analysis"""
        errors = []
        warnings = []
        
        try:
            # Check required methods
            if not hasattr(model, "predict"):
                errors.append("Model must have a 'predict' method")
            
            # Check for probability prediction capability
            if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
                warnings.append("Model doesn't support probability predictions. Some fairness metrics may be limited")
            
            # Check model type
            model_type = type(model).__name__
            supported_types = [
                'LogisticRegression', 'RandomForestClassifier', 'SVC', 'XGBClassifier',
                'LGBMClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier',
                'AdaBoostClassifier', 'GradientBoostingClassifier', 'MLPClassifier'
            ]
            
            if model_type not in supported_types:
                warnings.append(f"Model type '{model_type}' may not be fully supported. Proceed with caution")
            
            # Try to get model parameters
            try:
                if hasattr(model, "get_params"):
                    params = model.get_params()
                    if not params:
                        warnings.append("Could not retrieve model parameters")
            except Exception:
                warnings.append("Could not access model parameters")
            
            # Check if model is fitted
            try:
                if hasattr(model, "classes_"):
                    # Classification model
                    if len(model.classes_) < 2:
                        errors.append("Model appears to have fewer than 2 classes")
                    elif len(model.classes_) > 10:
                        warnings.append(f"Model has {len(model.classes_)} classes. Multi-class fairness analysis may be complex")
                elif hasattr(model, "coef_"):
                    # Linear model
                    if model.coef_ is None:
                        errors.append("Model appears to be unfitted")
            except Exception:
                warnings.append("Could not determine if model is properly fitted")
            
            is_valid = len(errors) == 0
            
            logger.info(f"Model validation completed: {model_type}, "
                       f"{len(errors)} errors, {len(warnings)} warnings")
            
            return ValidationResult(isValid=is_valid, errors=errors, warnings=warnings)
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return ValidationResult(
                isValid=False, 
                errors=[f"Validation error: {str(e)}"], 
                warnings=warnings
            )
    
    def validate_target_column(self, df: pd.DataFrame, target_column: str) -> ValidationResult:
        """Validate target column for fairness analysis"""
        errors = []
        warnings = []
        
        try:
            if target_column not in df.columns:
                errors.append(f"Target column '{target_column}' not found in dataset")
                return ValidationResult(isValid=False, errors=errors, warnings=warnings)
            
            target_series = df[target_column]
            
            # Check for missing values in target
            missing_count = target_series.isnull().sum()
            if missing_count > 0:
                errors.append(f"Target column has {missing_count} missing values")
            
            # Check number of unique values
            unique_count = target_series.nunique()
            if unique_count < 2:
                errors.append(f"Target column has only {unique_count} unique value(s)")
            elif unique_count > 20:
                warnings.append(f"Target column has {unique_count} unique values. Consider if this should be a regression problem")
            
            # Check class balance (for classification)
            if unique_count <= 10:
                value_counts = target_series.value_counts()
                min_class_size = value_counts.min()
                max_class_size = value_counts.max()
                
                if min_class_size / max_class_size < 0.1:  # Severe imbalance
                    warnings.append("Severe class imbalance detected. This may affect fairness analysis")
                elif min_class_size / max_class_size < 0.3:  # Moderate imbalance
                    warnings.append("Class imbalance detected. Consider this in your analysis")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(isValid=is_valid, errors=errors, warnings=warnings)
            
        except Exception as e:
            logger.error(f"Target column validation failed: {str(e)}")
            return ValidationResult(
                isValid=False, 
                errors=[f"Validation error: {str(e)}"], 
                warnings=warnings
            )
    
    def validate_sensitive_attributes(self, df: pd.DataFrame, sensitive_attributes: List[str]) -> ValidationResult:
        """Validate sensitive attributes for fairness analysis"""
        errors = []
        warnings = []
        
        try:
            for attr in sensitive_attributes:
                if attr not in df.columns:
                    errors.append(f"Sensitive attribute '{attr}' not found in dataset")
                    continue
                
                attr_series = df[attr]
                
                # Check for missing values
                missing_count = attr_series.isnull().sum()
                if missing_count > len(df) * 0.1:  # More than 10% missing
                    warnings.append(f"Sensitive attribute '{attr}' has {missing_count} missing values")
                
                # Check number of groups
                unique_count = attr_series.nunique()
                if unique_count < 2:
                    warnings.append(f"Sensitive attribute '{attr}' has only {unique_count} group(s)")
                elif unique_count > 20:
                    warnings.append(f"Sensitive attribute '{attr}' has {unique_count} groups. Analysis may be complex")
                
                # Check group sizes
                if unique_count >= 2:
                    group_sizes = attr_series.value_counts()
                    min_group_size = group_sizes.min()
                    
                    if min_group_size < 10:
                        warnings.append(f"Sensitive attribute '{attr}' has groups with very few samples (min: {min_group_size})")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(isValid=is_valid, errors=errors, warnings=warnings)
            
        except Exception as e:
            logger.error(f"Sensitive attributes validation failed: {str(e)}")
            return ValidationResult(
                isValid=False, 
                errors=[f"Validation error: {str(e)}"], 
                warnings=warnings
            )
