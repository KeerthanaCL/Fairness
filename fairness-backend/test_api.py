"""
Simple test script to verify FastAPI backend functionality
"""

import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:8000/api"

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_config_endpoints():
    """Test configuration endpoints"""
    try:
        # Test sensitive attributes
        response = requests.get(f"{BASE_URL}/config/sensitive-attributes")
        print(f"Sensitive Attributes: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data['attributes'])} sensitive attributes")
        
        # Test mitigation options
        response = requests.get(f"{BASE_URL}/config/mitigation-options")
        print(f"Mitigation Options: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data['options'])} mitigation options")
        
        return True
    except Exception as e:
        print(f"Config endpoints test failed: {e}")
        return False

def test_explanation_status():
    """Test explanation service status"""
    try:
        response = requests.get(f"{BASE_URL}/explanations/status")
        print(f"Explanation Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"AI Explanations Enabled: {data['enabled']}")
        return True
    except Exception as e:
        print(f"Explanation status test failed: {e}")
        return False

def test_upload_status():
    """Test upload status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/upload/status")
        print(f"Upload Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Upload Status: {data}")
        return True
    except Exception as e:
        print(f"Upload status test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing FastAPI Backend for Fairness Evaluation Platform")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Configuration Endpoints", test_config_endpoints),
        ("Explanation Service", test_explanation_status),
        ("Upload Status", test_upload_status)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
        print(f"✅ PASSED" if success else "❌ FAILED")
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print("-" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Backend is ready for frontend integration.")
        print("\nNext steps:")
        print("1. Update frontend .env: REACT_APP_USE_MOCK_DATA=false")
        print("2. Set API URL: REACT_APP_API_BASE_URL=http://localhost:8000/api")
        print("3. Start frontend: npm start")
    else:
        print("\n⚠️  Some tests failed. Check the backend logs and configuration.")

if __name__ == "__main__":
    main()
