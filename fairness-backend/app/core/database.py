"""
Database configuration and session management
"""

from sqlalchemy import create_engine, MetaData, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import asyncio
from typing import AsyncGenerator
import sqlite3
import json
from datetime import datetime

from app.core.config import settings

# Create SQLAlchemy engine
if settings.DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(settings.DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
metadata = MetaData()


async def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# In-memory storage for analysis sessions (for production, use Redis or database)
analysis_sessions = {}
file_storage = {}


# SQLite database connection helpers
def get_db_connection():
    """Get direct SQLite connection for custom operations"""
    return sqlite3.connect("fairness_platform.db")


def create_tables():
    """Create necessary database tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create feature transformations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feature_transformations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id TEXT NOT NULL,
        original_feature TEXT NOT NULL,
        transformed_feature TEXT NOT NULL,
        transformation_type TEXT NOT NULL,
        metadata_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create sensitive features metadata table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sensitive_features_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        feature_type TEXT NOT NULL,
        sensitivity_score REAL,
        p_value REAL,
        categories TEXT,
        is_sensitive BOOLEAN,
        detection_method TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(dataset_id, feature_name)
    )
    ''')
    
    # Create dataset metadata table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dataset_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id TEXT NOT NULL UNIQUE,
        original_dataset_id TEXT,
        is_transformed BOOLEAN DEFAULT FALSE,
        transformation_applied TEXT,
        columns_json TEXT,
        target_column TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()


def store_feature_transformation(dataset_id: str, original_feature: str, 
                                 transformed_feature: str, transformation_type: str, 
                                 metadata: dict = None):
    """Store feature transformation information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    metadata_json = json.dumps(metadata) if metadata else None
    
    cursor.execute('''
    INSERT INTO feature_transformations 
    (dataset_id, original_feature, transformed_feature, transformation_type, metadata_json)
    VALUES (?, ?, ?, ?, ?)
    ''', (dataset_id, original_feature, transformed_feature, transformation_type, metadata_json))
    
    conn.commit()
    conn.close()


def get_feature_transformation_metadata(dataset_id: str):
    """Get feature transformation metadata for a dataset"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT original_feature, transformed_feature, transformation_type, metadata_json
    FROM feature_transformations
    WHERE dataset_id = ?
    ''', (dataset_id,))
    
    transformations = {}
    for row in cursor.fetchall():
        original_feature, transformed_feature, transformation_type, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else {}
        transformations[original_feature] = {
            'transformed_feature': transformed_feature,
            'transformation_type': transformation_type,
            'metadata': metadata
        }
    
    conn.close()
    return transformations


def store_sensitive_features_metadata(dataset_id: str, features_metadata: list):
    """Store sensitive features metadata"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for feature_meta in features_metadata:
        cursor.execute('''
        INSERT OR REPLACE INTO sensitive_features_metadata 
        (dataset_id, feature_name, feature_type, sensitivity_score, p_value, 
         categories, is_sensitive, detection_method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id,
            feature_meta.get('name'),
            feature_meta.get('dataType'),
            feature_meta.get('sensitivityScore'),
            feature_meta.get('pValue'),
            json.dumps(feature_meta.get('categories', [])),
            feature_meta.get('isSensitive', False),
            feature_meta.get('detectionMethod', 'HSIC')
        ))
    
    conn.commit()
    conn.close()


def get_sensitive_features_metadata(dataset_id: str):
    """Get stored sensitive features metadata"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT feature_name, feature_type, sensitivity_score, p_value, 
           categories, is_sensitive, detection_method
    FROM sensitive_features_metadata
    WHERE dataset_id = ?
    ''', (dataset_id,))
    
    features = []
    for row in cursor.fetchall():
        feature_name, feature_type, sensitivity_score, p_value, categories_json, is_sensitive, detection_method = row
        categories = json.loads(categories_json) if categories_json else []
        
        features.append({
            'name': feature_name,
            'dataType': feature_type,
            'sensitivityScore': sensitivity_score,
            'pValue': p_value,
            'categories': categories,
            'isSensitive': is_sensitive,
            'detectionMethod': detection_method
        })
    
    conn.close()
    return features


def store_dataset_metadata(dataset_id: str, original_dataset_id: str = None, 
                          is_transformed: bool = False, transformation_applied: str = None,
                          columns: list = None, target_column: str = None):
    """Store dataset metadata"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    columns_json = json.dumps(columns) if columns else None
    
    cursor.execute('''
    INSERT OR REPLACE INTO dataset_metadata 
    (dataset_id, original_dataset_id, is_transformed, transformation_applied, columns_json, target_column)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (dataset_id, original_dataset_id, is_transformed, transformation_applied, columns_json, target_column))
    
    conn.commit()
    conn.close()


def get_dataset_metadata(dataset_id: str):
    """Get dataset metadata"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT original_dataset_id, is_transformed, transformation_applied, columns_json, target_column
    FROM dataset_metadata
    WHERE dataset_id = ?
    ''', (dataset_id,))
    
    row = cursor.fetchone()
    if row:
        original_dataset_id, is_transformed, transformation_applied, columns_json, target_column = row
        columns = json.loads(columns_json) if columns_json else []
        
        conn.close()
        return {
            'original_dataset_id': original_dataset_id,
            'is_transformed': is_transformed,
            'transformation_applied': transformation_applied,
            'columns': columns,
            'target_column': target_column
        }
    
    conn.close()
    return None
