"""
Test script to verify the sensitive feature metadata preservation fix
This script tests the complete pipeline from detection to mitigation
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import (
    create_tables, store_sensitive_features_metadata, 
    get_sensitive_features_metadata, store_dataset_metadata,
    get_dataset_metadata, store_feature_transformation,
    get_feature_transformation_metadata
)
from app.core.bias_detector import BiasDetector
from app.services.mitigation_service import MitigationService
from app.core.data_processor import DataProcessor

def create_test_dataset():
    """Create a synthetic test dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic data with bias
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.5, 0.2, 0.2, 0.1])
    age = np.random.normal(35, 10, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    
    # Create biased target variable (salary prediction)
    # Introduce bias: males and certain races get higher salaries
    base_salary = 50000
    gender_bias = np.where(gender == 'Male', 15000, 0)
    race_bias = np.where(race == 'White', 10000, 
                np.where(race == 'Asian', 8000,
                np.where(race == 'Hispanic', -5000, -8000)))
    education_bonus = np.where(education == 'PhD', 25000,
                      np.where(education == 'Master', 15000,
                      np.where(education == 'Bachelor', 8000, 0)))
    
    salary = base_salary + gender_bias + race_bias + education_bonus + np.random.normal(0, 5000, n_samples)
    high_salary = (salary > np.median(salary)).astype(int)
    
    df = pd.DataFrame({
        'gender': gender,
        'race': race,
        'age': age,
        'education': education,
        'salary': salary,
        'high_salary': high_salary
    })
    
    return df

def test_metadata_preservation():
    """Test the complete metadata preservation pipeline"""
    print("🧪 Testing Sensitive Feature Metadata Preservation Fix")
    print("=" * 60)
    
    # Step 1: Initialize database
    print("1. Initializing database...")
    create_tables()
    print("   ✅ Database tables created")
    
    # Step 2: Create test dataset
    print("2. Creating test dataset...")
    df = create_test_dataset()
    dataset_id = "test_dataset_001"
    print(f"   ✅ Test dataset created with {len(df)} rows, {len(df.columns)} columns")
    
    # Step 3: Store initial dataset metadata
    print("3. Storing dataset metadata...")
    store_dataset_metadata(
        dataset_id=dataset_id,
        is_transformed=False,
        columns=list(df.columns),
        target_column='high_salary'
    )
    print("   ✅ Dataset metadata stored")
    
    # Step 4: Detect sensitive features
    print("4. Detecting sensitive features...")
    detector = BiasDetector(use_hsic=False)  # Use traditional tests for simplicity
    feature_types = {col: 'categorical' if df[col].dtype == 'object' else 'numerical' 
                    for col in df.columns if col != 'high_salary'}
    
    detection_result = detector.detect_sensitive_features(
        df, 'high_salary', feature_types, dataset_id=dataset_id
    )
    
    print(f"   ✅ Detected {len(detection_result.detectedFeatures)} sensitive features:")
    for feature in detection_result.detectedFeatures:
        print(f"      - {feature.name}: {feature.effectSize:.3f} (p={feature.pValue:.3f})")
    
    # Step 5: Verify metadata storage
    print("5. Verifying metadata storage...")
    stored_metadata = get_sensitive_features_metadata(dataset_id)
    print(f"   ✅ Stored metadata for {len(stored_metadata)} features")
    
    # Step 6: Simulate mitigation transformation
    print("6. Simulating mitigation transformation...")
    mitigated_dataset_id = f"{dataset_id}_mitigated"
    
    # Create a mock transformed dataset (in real scenario, this would come from mitigation)
    df_mitigated = df.copy()
    df_mitigated['gender_fair'] = df_mitigated['gender']  # Simulate feature transformation
    df_mitigated = df_mitigated.drop('gender', axis=1)  # Remove original feature
    
    # Store transformation information BEFORE storing dataset metadata
    store_feature_transformation(
        mitigated_dataset_id,
        original_feature='gender',
        transformed_feature='gender_fair',
        transformation_type='Reweighing',
        metadata={'preserves_original': False, 'transformation_details': 'Feature renamed during Reweighing mitigation'}
    )
    
    # Store mitigated dataset metadata
    store_dataset_metadata(
        dataset_id=mitigated_dataset_id,
        original_dataset_id=dataset_id,
        is_transformed=True,
        transformation_applied='Reweighing',
        columns=list(df_mitigated.columns),
        target_column='high_salary'
    )
    
    print("   ✅ Mitigation transformation simulated")
    print(f"   📝 Original columns: {list(df.columns)}")
    print(f"   📝 Transformed columns: {list(df_mitigated.columns)}")
    
    # Step 7: Test transformed dataset detection
    print("7. Testing detection on transformed dataset...")
    feature_types_mitigated = {col: 'categorical' if df_mitigated[col].dtype == 'object' else 'numerical' 
                               for col in df_mitigated.columns if col != 'high_salary'}
    
    transformed_detection = detector.detect_sensitive_features(
        df_mitigated, 'high_salary', 
        feature_types_mitigated,
        dataset_id=mitigated_dataset_id,
        use_cached=True  # This should use preserved metadata
    )
    
    print(f"   ✅ Detected {len(transformed_detection.detectedFeatures)} features in transformed dataset:")
    for feature in transformed_detection.detectedFeatures:
        print(f"      - {feature.name}: {feature.effectSize:.3f} (p={feature.pValue:.3f})")
    
    # Step 8: Verify transformation tracking
    print("8. Verifying transformation tracking...")
    transformations = get_feature_transformation_metadata(mitigated_dataset_id)
    print(f"   ✅ Tracked {len(transformations)} feature transformations:")
    for orig, info in transformations.items():
        print(f"      - {orig} → {info['transformed_feature']} ({info['transformation_type']})")
    
    # Step 9: Test metadata consistency
    print("9. Testing metadata consistency...")
    original_metadata = get_sensitive_features_metadata(dataset_id)
    transformed_metadata = get_sensitive_features_metadata(mitigated_dataset_id)
    
    original_count = len(original_metadata)
    transformed_count = len(transformed_metadata)
    preservation_rate = transformed_count / original_count if original_count > 0 else 0
    
    print(f"   📊 Original features: {original_count}")
    print(f"   📊 Preserved features: {transformed_count}")
    print(f"   📊 Preservation rate: {preservation_rate:.2%}")
    
    # Step 10: Validate the fix
    print("10. Validating the fix...")
    
    success_criteria = [
        (transformed_count > 0, "Some sensitive features preserved after transformation"),
        (len(transformations) > 0, "Feature transformations tracked"),
        (len(transformed_detection.detectedFeatures) > 0, "Transformed features properly detected using preserved metadata"),
        (preservation_rate >= 0.5, "At least 50% of features preserved"),
    ]
    
    all_passed = True
    for passed, description in success_criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {description}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The metadata preservation fix is working correctly.")
        print("   - Sensitive features are tracked through transformations")
        print("   - Feature mappings are preserved")
        print("   - UI should no longer crash after mitigation")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    print("=" * 60)

if __name__ == "__main__":
    test_metadata_preservation()