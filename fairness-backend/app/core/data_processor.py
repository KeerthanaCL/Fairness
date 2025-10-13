"""
Data processing utilities for fairness analysis
Enhanced version adapted from backend_old with improved error handling
Enhanced with metadata preservation capabilities
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.base import BaseEstimator
import uuid

from app.core.config import settings
from app.core.database import (
    store_dataset_metadata, get_dataset_metadata,
    store_sensitive_features_metadata, get_sensitive_features_metadata
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """Enhanced data processing for fairness analysis"""
    
    def __init__(self):
        self.supported_formats = {
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.parquet': self._load_parquet
        }
        
    def load_dataset(self, file_path: Union[str, Path], dataset_id: str = None) -> pd.DataFrame:
        """
        Load dataset from various formats with robust error handling
        Enhanced with metadata tracking
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        try:
            loader_func = self.supported_formats[file_extension]
            df = loader_func(file_path)
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Generate dataset_id if not provided
            if not dataset_id:
                dataset_id = str(uuid.uuid4())
            
            # Store dataset metadata
            store_dataset_metadata(
                dataset_id=dataset_id,
                is_transformed=False,
                columns=list(df.columns)
            )
            
            logger.info(f"Successfully loaded dataset: {file_path.name} with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset {file_path}: {str(e)}")
            raise
    
    def save_dataset_with_metadata(
        self, 
        df: pd.DataFrame, 
        file_path: Union[str, Path], 
        dataset_id: str,
        original_dataset_id: str = None,
        transformation_applied: str = None,
        preserve_metadata: bool = True
    ) -> str:
        """
        Save dataset with metadata preservation
        
        Args:
            df: DataFrame to save
            file_path: Path to save the dataset
            dataset_id: Unique identifier for this dataset
            original_dataset_id: ID of the original dataset (if this is a transformation)
            transformation_applied: Description of transformation applied
            preserve_metadata: Whether to preserve sensitive feature metadata
            
        Returns:
            Dataset ID of the saved dataset
        """
        file_path = Path(file_path)
        
        try:
            # Save the dataset
            if file_path.suffix.lower() == '.csv':
                df.to_csv(file_path, index=False)
            elif file_path.suffix.lower() == '.parquet':
                df.to_parquet(file_path, index=False)
            elif file_path.suffix.lower() == '.pickle':
                df.to_pickle(file_path)
            else:
                # Default to CSV
                file_path = file_path.with_suffix('.csv')
                df.to_csv(file_path, index=False)
            
            # Store dataset metadata
            store_dataset_metadata(
                dataset_id=dataset_id,
                original_dataset_id=original_dataset_id,
                is_transformed=original_dataset_id is not None,
                transformation_applied=transformation_applied,
                columns=list(df.columns)
            )
            
            # Preserve sensitive features metadata if requested and available
            if preserve_metadata and original_dataset_id:
                self._preserve_sensitive_features_metadata(
                    original_dataset_id, dataset_id, df.columns
                )
            
            logger.info(f"Successfully saved dataset with metadata: {file_path}")
            return dataset_id
            
        except Exception as e:
            logger.error(f"Failed to save dataset with metadata: {str(e)}")
            raise
    
    def _preserve_sensitive_features_metadata(
        self, 
        original_dataset_id: str, 
        new_dataset_id: str, 
        current_columns: List[str]
    ):
        """Preserve sensitive features metadata from original dataset"""
        
        # Get original sensitive features metadata
        original_features = get_sensitive_features_metadata(original_dataset_id)
        
        if not original_features:
            return
        
        # Filter features that still exist in the transformed dataset
        preserved_features = []
        current_columns_set = set(current_columns)
        
        for feature_meta in original_features:
            if feature_meta['name'] in current_columns_set:
                # Feature still exists, preserve its metadata
                preserved_feature = feature_meta.copy()
                preserved_feature['transformation_preserved'] = True
                preserved_features.append(preserved_feature)
            else:
                logger.warning(f"Sensitive feature {feature_meta['name']} not found in transformed dataset")
        
        # Store preserved metadata for the new dataset
        if preserved_features:
            store_sensitive_features_metadata(new_dataset_id, preserved_features)
            logger.info(f"Preserved metadata for {len(preserved_features)} sensitive features")
    
    def get_dataset_with_metadata(self, dataset_id: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Get dataset along with its metadata
        
        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        try:
            # Get dataset metadata
            metadata = get_dataset_metadata(dataset_id)
            
            if not metadata:
                logger.warning(f"No metadata found for dataset {dataset_id}")
                return None, {}
            
            # Try to load the dataset (this would need to be adapted based on how you store datasets)
            # For now, return the metadata
            return None, metadata
            
        except Exception as e:
            logger.error(f"Failed to get dataset with metadata: {str(e)}")
            return None, {}
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with automatic encoding detection"""
        try:
            # Try UTF-8 first
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Try latin-1 encoding
                return pd.read_csv(file_path, encoding='latin-1')
            except Exception:
                # Try with automatic encoding detection
                return pd.read_csv(file_path, encoding='ISO-8859-1')
    
    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Load JSON file"""
        return pd.read_json(file_path)
    
    def _load_excel(self, file_path: Path) -> pd.DataFrame:
        """Load Excel file"""
        return pd.read_excel(file_path)
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load Parquet file"""
        return pd.read_parquet(file_path)
    
    def load_model(self, file_path: Union[str, Path]) -> BaseEstimator:
        """
        Load ML model from .pkl or .joblib formats with robust error handling
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in settings.SUPPORTED_MODEL_FORMATS:
            raise ValueError(f"Unsupported model format: {file_extension}. Supported formats: {settings.SUPPORTED_MODEL_FORMATS}")
        
        try:
            if file_extension == '.joblib':
                model = joblib.load(file_path)
            elif file_extension == '.pkl':
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported model format: {file_extension}")
            
            # Validate model has required methods
            if not hasattr(model, 'predict'):
                raise ValueError("Model must have a 'predict' method")
            
            logger.info(f"Successfully loaded model: {file_path.name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {file_path}: {str(e)}")
            raise
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive dataset validation with detailed reporting
        """
        validation_result = {
            "isValid": True,
            "errors": [],
            "warnings": [],
            "shape": df.shape,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": {}
        }
        
        try:
            # Check for empty dataset
            if df.empty:
                validation_result["errors"].append("Dataset is empty")
                validation_result["isValid"] = False
                return validation_result
            
            # Check for minimum rows
            if len(df) < 10:
                validation_result["warnings"].append(f"Dataset has only {len(df)} rows. Minimum 10 recommended.")
            
            # Check for missing values
            missing_percentage = (df.isnull().sum() / len(df) * 100)
            high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
            
            if high_missing_cols:
                validation_result["warnings"].append(
                    f"Columns with >50% missing values: {high_missing_cols}"
                )
            
            # Generate summary statistics for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                validation_result["summary_stats"]["numerical"] = (
                    df[numerical_cols].describe().to_dict()
                )
            
            # Generate summary for categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                categorical_summary = {}
                for col in categorical_cols:
                    categorical_summary[col] = {
                        "unique_values": df[col].nunique(),
                        "top_values": df[col].value_counts().head(5).to_dict()
                    }
                validation_result["summary_stats"]["categorical"] = categorical_summary
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                validation_result["warnings"].append(f"Found {duplicate_count} duplicate rows")
            
            # Check for constant columns
            constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
            if constant_cols:
                validation_result["warnings"].append(f"Constant columns detected: {constant_cols}")
            
            logger.info(f"Dataset validation completed. Valid: {validation_result['isValid']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            validation_result["errors"].append(f"Validation failed: {str(e)}")
            validation_result["isValid"] = False
            return validation_result
    
    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect feature types for statistical testing
        """
        feature_types = {}
        
        for column in df.columns:
            col_data = df[column]
            
            # Skip if mostly null
            if col_data.isnull().sum() / len(col_data) > 0.8:
                feature_types[column] = "invalid"
                continue
            
            # Determine type based on dtype and unique values
            if col_data.dtype in ['object', 'category']:
                feature_types[column] = "categorical"
            elif col_data.dtype in ['bool']:
                feature_types[column] = "binary"
            else:
                # Numerical column - check if it should be treated as categorical
                unique_values = col_data.nunique()
                total_values = len(col_data.dropna())
                
                if unique_values <= 10 or unique_values / total_values < 0.05:
                    feature_types[column] = "categorical"
                elif unique_values == 2:
                    feature_types[column] = "binary"
                else:
                    feature_types[column] = "numerical"
        
        return feature_types
    
    def preprocess_for_analysis(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess dataset for fairness analysis
        """
        df_processed = df.copy()
        preprocessing_info = {
            "original_shape": df.shape,
            "transformations": [],
            "removed_columns": [],
            "removed_rows": 0
        }
        
        try:
            # Remove rows where target column is null
            initial_rows = len(df_processed)
            df_processed = df_processed.dropna(subset=[target_column])
            removed_rows = initial_rows - len(df_processed)
            preprocessing_info["removed_rows"] = removed_rows
            
            if removed_rows > 0:
                preprocessing_info["transformations"].append(f"Removed {removed_rows} rows with null target values")
            
            # Remove completely empty columns
            empty_cols = df_processed.columns[df_processed.isnull().all()].tolist()
            if empty_cols:
                df_processed = df_processed.drop(columns=empty_cols)
                preprocessing_info["removed_columns"].extend(empty_cols)
                preprocessing_info["transformations"].append(f"Removed empty columns: {empty_cols}")
            
            # Convert boolean columns to string for consistency
            bool_cols = df_processed.select_dtypes(include=['bool']).columns
            for col in bool_cols:
                df_processed[col] = df_processed[col].astype(str)
                preprocessing_info["transformations"].append(f"Converted boolean column {col} to string")
            
            preprocessing_info["final_shape"] = df_processed.shape
            
            logger.info(f"Preprocessing completed. Shape: {df.shape} -> {df_processed.shape}")
            return df_processed, preprocessing_info
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def validate_model_compatibility(self, model: BaseEstimator, feature_columns: List[str]) -> Dict[str, Any]:
        """
        Validate model compatibility with dataset features
        """
        validation_result = {
            "isCompatible": True,
            "errors": [],
            "warnings": [],
            "model_info": {}
        }
        
        try:
            # Get model information
            model_type = type(model).__name__
            validation_result["model_info"]["type"] = model_type
            
            # Check if model has required methods
            required_methods = ['predict']
            for method in required_methods:
                if not hasattr(model, method):
                    validation_result["errors"].append(f"Model missing required method: {method}")
                    validation_result["isCompatible"] = False
            
            # Check for predict_proba method (useful for fairness analysis)
            if hasattr(model, 'predict_proba'):
                validation_result["model_info"]["has_predict_proba"] = True
            else:
                validation_result["warnings"].append("Model doesn't have predict_proba method. Some fairness metrics may be limited.")
                validation_result["model_info"]["has_predict_proba"] = False
            
            # Try to get feature importance if available
            if hasattr(model, 'feature_importances_'):
                validation_result["model_info"]["has_feature_importance"] = True
                if len(model.feature_importances_) != len(feature_columns):
                    validation_result["warnings"].append(
                        f"Feature importance length ({len(model.feature_importances_)}) doesn't match feature count ({len(feature_columns)})"
                    )
            else:
                validation_result["model_info"]["has_feature_importance"] = False
            
            logger.info(f"Model validation completed. Compatible: {validation_result['isCompatible']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            validation_result["errors"].append(f"Validation failed: {str(e)}")
            validation_result["isCompatible"] = False
            return validation_result
    
    def get_data_preview(self, df: pd.DataFrame, num_rows: int = 5) -> Dict[str, Any]:
        """
        Generate data preview for frontend display
        """
        try:
            preview = {
                "rows": len(df),
                "columns": len(df.columns),
                "sampleData": df.head(num_rows).fillna("null").to_dict(orient="records")
            }
            return preview
        except Exception as e:
            logger.error(f"Failed to generate data preview: {str(e)}")
            return {"rows": 0, "columns": 0, "sampleData": []}
