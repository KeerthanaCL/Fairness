"""
File handling utilities for upload management
Supports both .pkl and .joblib model formats
"""

import os
import pandas as pd
import joblib
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
from fastapi import UploadFile, HTTPException
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file operations for uploads and model loading"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        
    def validate_file(self, file: UploadFile, allowed_types: List[str]) -> bool:
        """Validate uploaded file"""
        # Check file size
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
            )
        
        return True
    
    async def save_file(self, file: UploadFile, subdirectory: str, file_type: str) -> Path:
        """Save uploaded file to designated directory"""
        try:
            # Create subdirectory if it doesn't exist
            save_dir = self.upload_dir / subdirectory
            save_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            file_extension = Path(file.filename).suffix.lower()
            unique_filename = f"{file_type}_{uuid.uuid4().hex}{file_extension}"
            file_path = save_dir / unique_filename
            
            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            logger.info(f"File saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")
    
    def load_dataset(self, file_path: Path) -> pd.DataFrame:
        """Load dataset from file"""
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            elif file_extension == ".json":
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported dataset format: {file_extension}")
            
            logger.info(f"Dataset loaded: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Dataset load failed: {str(e)}")
    
    def load_model(self, file_path: Path) -> Any:
        """Load ML model from file (supports both .pkl and .joblib)"""
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == ".joblib":
                model = joblib.load(file_path)
            elif file_extension == ".pkl":
                with open(file_path, "rb") as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported model format: {file_extension}")
            
            # Validate model has required methods
            if not hasattr(model, "predict"):
                raise ValueError("Model must have a 'predict' method")
            
            logger.info(f"Model loaded: {file_path} (type: {type(model).__name__})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Model load failed: {str(e)}")
    
    def extract_model_info(self, model: Any) -> Dict[str, Any]:
        """Extract information about the loaded model"""
        try:
            model_info = {
                "type": type(model).__name__,
                "algorithm": model.__class__.__name__,
                "has_predict": hasattr(model, "predict"),
                "has_predict_proba": hasattr(model, "predict_proba"),
                "has_decision_function": hasattr(model, "decision_function")
            }
            
            # Try to get number of features
            if hasattr(model, "n_features_in_"):
                model_info["features"] = model.n_features_in_
            elif hasattr(model, "coef_"):
                if hasattr(model.coef_, "shape"):
                    model_info["features"] = model.coef_.shape[-1]
            else:
                model_info["features"] = "Unknown"
            
            # Try to get model parameters
            try:
                if hasattr(model, "get_params"):
                    model_info["parameters"] = model.get_params()
                else:
                    # For models without get_params, try to get basic attributes
                    model_info["parameters"] = {
                        "class_name": model.__class__.__name__,
                        "module": model.__class__.__module__
                    }
            except Exception as param_e:
                logger.debug(f"Could not extract model parameters: {str(param_e)}")
                model_info["parameters"] = {
                    "class_name": model.__class__.__name__,
                    "extraction_error": str(param_e)
                }
            
            return model_info
            
        except Exception as e:
            logger.warning(f"Failed to extract model info: {str(e)}")
            return {
                "type": "Unknown",
                "algorithm": "Unknown",
                "features": 0,
                "has_predict": hasattr(model, "predict"),
                "has_predict_proba": hasattr(model, "predict_proba")
            }
    
    def cleanup_files(self, subdirectory: str, file_type: str) -> bool:
        """Clean up files of a specific type"""
        try:
            cleanup_dir = self.upload_dir / subdirectory
            if not cleanup_dir.exists():
                return True
            
            # Find and remove files matching the pattern
            pattern = f"{file_type}_*"
            removed_count = 0
            
            for file_path in cleanup_dir.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} files in {cleanup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup files: {str(e)}")
            return False
    
    def check_file_exists(self, subdirectory: str, file_type: str) -> bool:
        """Check if files of a specific type exist"""
        try:
            check_dir = self.upload_dir / subdirectory
            if not check_dir.exists():
                return False
            
            pattern = f"{file_type}_*"
            existing_files = list(check_dir.glob(pattern))
            
            return len(existing_files) > 0
            
        except Exception as e:
            logger.error(f"Failed to check file existence: {str(e)}")
            return False
    
    def get_latest_file(self, subdirectory: str, file_type: str) -> Optional[Path]:
        """Get the most recently uploaded file of a specific type"""
        try:
            check_dir = self.upload_dir / subdirectory
            if not check_dir.exists():
                return None
            
            pattern = f"{file_type}_*"
            files = list(check_dir.glob(pattern))
            
            if not files:
                return None
            
            # Return the most recently modified file
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            return latest_file
            
        except Exception as e:
            logger.error(f"Failed to get latest file: {str(e)}")
            return None
    
    def get_file_stats(self, file_path: Path) -> Dict[str, Any]:
        """Get file statistics"""
        try:
            stats = file_path.stat()
            return {
                "size": stats.st_size,
                "created": stats.st_ctime,
                "modified": stats.st_mtime,
                "exists": file_path.exists(),
                "extension": file_path.suffix.lower()
            }
        except Exception as e:
            logger.error(f"Failed to get file stats: {str(e)}")
            return {}
