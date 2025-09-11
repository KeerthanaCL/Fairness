"""
File upload API endpoints
Handles training data, testing data, and model file uploads
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import pandas as pd
import joblib
import pickle
from pathlib import Path
import uuid
import logging

from app.models.schemas import UploadResponse, ValidationResult, DataPreview, UploadValidation
from app.core.config import settings
from app.utils.file_handler import FileHandler
from app.utils.validation import DataValidator

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize utilities
file_handler = FileHandler()
data_validator = DataValidator()

# In-memory upload registry (in production, this should be in a database)
upload_registry: Dict[str, Dict[str, Any]] = {}

def get_upload_info(upload_id: str) -> Optional[Dict[str, Any]]:
    """Get upload information by upload ID"""
    return upload_registry.get(upload_id)

def get_upload_file_path(upload_id: str) -> Optional[str]:
    """Get file path for an upload ID"""
    upload_info = get_upload_info(upload_id)
    return upload_info.get("file_path") if upload_info else None

@router.post("/training", response_model=UploadResponse)
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload training dataset
    Expected frontend call: POST /api/upload/training
    """
    try:
        # Validate file
        file_handler.validate_file(file, allowed_types=[".csv", ".json"])
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Save file
        file_path = await file_handler.save_file(file, "datasets", "training")
        
        # Load and validate dataset
        df = file_handler.load_dataset(file_path)
        validation_result = data_validator.validate_dataset(df, is_training=True)
        
        # Convert ValidationResult to UploadValidation
        validation = UploadValidation(
            isValid=validation_result.isValid,
            errors=validation_result.errors,
            warnings=validation_result.warnings
        )
        
        # Create preview
        preview = DataPreview(
            rows=len(df),
            sampleData=df.head(5).to_dict('records') if len(df) > 0 else []
        )
        
        # Store upload info in registry
        upload_registry[upload_id] = {
            "id": upload_id,
            "type": "training_dataset",
            "filename": file.filename,
            "file_path": str(file_path),
            "size": file.size,
            "columns": list(df.columns),
            "upload_time": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Training data uploaded successfully: {file.filename}")
        
        return UploadResponse(
            success=True,
            filename=file.filename,
            size=file.size,
            upload_id=upload_id,
            columns=list(df.columns),
            preview=preview,
            validation=validation
        )
        
    except Exception as e:
        logger.error(f"Training data upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@router.post("/testing", response_model=UploadResponse)
async def upload_testing_data(file: UploadFile = File(...)):
    """
    Upload testing dataset
    Expected frontend call: POST /api/upload/testing
    """
    try:
        # Validate file
        file_handler.validate_file(file, allowed_types=[".csv", ".json"])
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Save file
        file_path = await file_handler.save_file(file, "datasets", "testing")
        
        # Load and validate dataset
        df = file_handler.load_dataset(file_path)
        validation_result = data_validator.validate_dataset(df, is_training=False)
        
        # Convert ValidationResult to UploadValidation
        validation = UploadValidation(
            isValid=validation_result.isValid,
            errors=validation_result.errors,
            warnings=validation_result.warnings
        )
        
        # Create preview
        preview = DataPreview(
            rows=len(df),
            sampleData=df.head(5).to_dict('records') if len(df) > 0 else []
        )
        
        # Store upload info in registry
        upload_registry[upload_id] = {
            "id": upload_id,
            "type": "testing_dataset",
            "filename": file.filename,
            "file_path": str(file_path),
            "size": file.size,
            "columns": list(df.columns),
            "upload_time": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Testing data uploaded successfully: {file.filename}")
        
        return UploadResponse(
            success=True,
            filename=file.filename,
            size=file.size,
            upload_id=upload_id,
            columns=list(df.columns),
            preview=preview,
            validation=validation
        )
        
    except Exception as e:
        logger.error(f"Testing data upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@router.post("/model", response_model=UploadResponse)
async def upload_model_file(file: UploadFile = File(...)):
    """
    Upload ML model file (supports .pkl and .joblib formats)
    Expected frontend call: POST /api/upload/model
    """
    try:
        # Validate file
        file_handler.validate_file(file, allowed_types=[".pkl", ".joblib"])
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Save file
        file_path = await file_handler.save_file(file, "models", "model")
        
        # Load and validate model
        model = file_handler.load_model(file_path)
        validation_result = data_validator.validate_model(model)
        
        # Convert ValidationResult to UploadValidation
        validation = UploadValidation(
            isValid=validation_result.isValid,
            errors=validation_result.errors,
            warnings=validation_result.warnings
        )
        
        # Extract model information
        model_info = file_handler.extract_model_info(model)
        
        # Create preview with model information
        preview = DataPreview(
            rows=0,  # Models don't have rows
            sampleData=[{
                "model_type": model_info.get("type", "Unknown"),
                "features": model_info.get("features", 0),
                "algorithm": model_info.get("algorithm", "Unknown"),
                "has_predict": hasattr(model, "predict"),
                "has_predict_proba": hasattr(model, "predict_proba")
            }]
        )
        
        # Store upload info in registry
        upload_registry[upload_id] = {
            "id": upload_id,
            "type": "model",
            "filename": file.filename,
            "file_path": str(file_path),
            "size": file.size,
            "model_info": model_info,
            "upload_time": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Model uploaded successfully: {file.filename}")
        
        return UploadResponse(
            success=True,
            filename=file.filename,
            size=file.size,
            upload_id=upload_id,
            columns=[],  # Models don't have columns
            preview=preview,
            validation=validation
        )
        
    except Exception as e:
        logger.error(f"Model upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@router.delete("/training")
async def delete_training_data():
    """Delete uploaded training data"""
    try:
        file_handler.cleanup_files("datasets", "training")
        return {"success": True, "message": "Training data deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Delete failed: {str(e)}")

@router.delete("/testing")
async def delete_testing_data():
    """Delete uploaded testing data"""
    try:
        file_handler.cleanup_files("datasets", "testing")
        return {"success": True, "message": "Testing data deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Delete failed: {str(e)}")

@router.delete("/model")
async def delete_model():
    """Delete uploaded model"""
    try:
        file_handler.cleanup_files("models", "model")
        return {"success": True, "message": "Model deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Delete failed: {str(e)}")

@router.get("/status")
async def get_upload_status():
    """Get current upload status"""
    try:
        status = {
            "training": file_handler.check_file_exists("datasets", "training"),
            "testing": file_handler.check_file_exists("datasets", "testing"),
            "model": file_handler.check_file_exists("models", "model")
        }
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
