# Fairness Evaluation Platform - FastAPI Backend

A production-ready FastAPI backend for ML fairness analysis and bias detection, designed to integrate seamlessly with the React frontend.

## 🚀 Features

- **Complete API Integration**: Matches frontend data contracts exactly
- **Dual Model Support**: Both .pkl and .joblib model formats
- **Comprehensive Fairness Analysis**: Statistical bias detection and fairness metrics
- **Async Operations**: Background analysis processing with real-time status updates
- **Automatic Documentation**: OpenAPI/Swagger docs at `/api/docs`
- **Production Ready**: Structured for scalability and deployment

## 📁 Project Structure

```
fairness-backend/
├── app/
│   ├── api/                    # API endpoints
│   │   ├── upload.py          # File upload endpoints
│   │   ├── analysis.py        # Analysis management
│   │   ├── config.py          # Configuration endpoints
│   │   └── explanations.py    # AI explanations
│   ├── core/                   # Core analysis modules
│   │   ├── bias_detector.py   # Bias detection algorithms
│   │   ├── fairness_metrics.py # Fairness calculations
│   │   ├── data_processor.py  # Data loading and processing
│   │   ├── config.py          # Application settings
│   │   └── database.py        # Database configuration
│   ├── models/                 # Pydantic schemas
│   │   └── schemas.py         # API request/response models
│   ├── services/              # Business logic
│   │   ├── analysis_service.py # Analysis orchestration
│   │   └── explanation_service.py # AI explanations
│   └── utils/                 # Utilities
│       ├── file_handler.py    # File operations
│       └── validation.py      # Data validation
├── uploads/                    # File storage
│   ├── datasets/              # Training/testing data
│   └── models/                # ML model files
├── main.py                    # FastAPI application
├── requirements.txt           # Dependencies
└── .env                       # Configuration
```

## 🔧 Installation & Setup

### 1. Create Virtual Environment

```bash
cd fairness-backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

Copy `.env` file and adjust settings as needed:

```bash
# Server Settings
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Frontend URLs for CORS
ALLOWED_ORIGINS=["http://localhost:3000"]

# File Upload
MAX_FILE_SIZE=104857600  # 100MB
UPLOAD_DIR=uploads

# Database
DATABASE_URL=sqlite:///./fairness_platform.db
```

### 4. Run the Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🌐 API Endpoints

### File Upload
- `POST /api/upload/training` - Upload training dataset
- `POST /api/upload/testing` - Upload testing dataset  
- `POST /api/upload/model` - Upload ML model (.pkl/.joblib)

### Analysis Management
- `POST /api/analysis/start` - Start fairness analysis
- `GET /api/analysis/status/{id}` - Get analysis status
- `DELETE /api/analysis/{id}` - Stop analysis

### Results Retrieval
- `GET /api/analysis/sensitive-features/{id}` - Get detected sensitive features
- `GET /api/analysis/fairness-metrics/{id}` - Get fairness metrics
- `GET /api/analysis/mitigation-strategies/{id}` - Get mitigation recommendations
- `GET /api/analysis/before-after-comparison/{id}` - Get comparison results

### Configuration
- `GET /api/config/sensitive-attributes` - Available sensitive attributes
- `GET /api/config/mitigation-options` - Available mitigation strategies

### Health & Documentation
- `GET /api/health` - Health check
- `GET /api/docs` - Interactive API documentation
- `GET /api/redoc` - Alternative documentation

## 🎯 Frontend Integration

This backend is designed to work seamlessly with the React frontend:

### 1. Update Frontend Environment

```bash
# In fairness-evaluation-platform/.env
REACT_APP_API_BASE_URL=http://localhost:8000/api
REACT_APP_USE_MOCK_DATA=false
```

### 2. Start Both Services

```bash
# Terminal 1: Backend
cd fairness-backend
uvicorn main:app --reload

# Terminal 2: Frontend  
cd fairness-evaluation-platform
npm start
```

### 3. Verify Integration

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/api/docs
- Health Check: http://localhost:8000/api/health

## 📊 Supported Features

### Model Formats
- ✅ **Pickle (.pkl)**: Standard Python pickle format
- ✅ **Joblib (.joblib)**: Optimized for scikit-learn models
- ✅ **Auto-detection**: Automatic format recognition

### Dataset Formats
- ✅ **CSV**: Comma-separated values
- ✅ **JSON**: JavaScript Object Notation
- ✅ **Excel**: .xlsx and .xls files (optional)

### Fairness Metrics
- ✅ **Statistical Parity**: Demographic parity analysis
- ✅ **Disparate Impact**: 80% rule compliance
- ✅ **Equal Opportunity**: True positive rate equality
- ✅ **Equalized Odds**: TPR and FPR equality
- ✅ **Calibration**: Predictive parity analysis

### Bias Detection
- ✅ **Statistical Tests**: Chi-square, ANOVA, Pearson correlation
- ✅ **Automatic Detection**: Smart sensitive feature identification
- ✅ **Multi-attribute Support**: Multiple sensitive attributes
- ✅ **Effect Size Analysis**: Cohen's d, Cramér's V

### Mitigation Strategies
- ✅ **Preprocessing**: Reweighing, disparate impact remover
- ✅ **In-processing**: Adversarial debiasing, fair classification
- ✅ **Post-processing**: Calibrated equalized odds, threshold optimization

## 🔒 Security Features

### File Upload Security
- File type validation (.pkl, .joblib, .csv, .json)
- File size limits (configurable, default 100MB)
- Secure file storage with unique naming
- Path traversal protection

### API Security
- CORS protection for frontend integration
- Request validation with Pydantic models
- Error handling without information leakage
- Input sanitization

### Data Privacy
- Temporary file storage with cleanup
- In-memory analysis processing
- No persistent storage of sensitive data
- Configurable data retention policies

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production

```bash
DEBUG=false
HOST=0.0.0.0
PORT=8000
DATABASE_URL=postgresql://user:password@localhost/fairness_db
ALLOWED_ORIGINS=["https://your-frontend-domain.com"]
```

## 📈 Performance

### Async Processing
- Background analysis processing
- Non-blocking file uploads
- Real-time progress tracking
- Concurrent request handling

### Scalability
- Stateless design for horizontal scaling
- Database-backed session storage
- Configurable resource limits
- Background task queuing support

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/

# Test specific module
pytest tests/test_upload.py
```

## 📝 Development

### Adding New Endpoints

1. Define Pydantic models in `app/models/schemas.py`
2. Create endpoint in appropriate `app/api/*.py` file
3. Add business logic in `app/services/`
4. Update router registration in `main.py`

### Adding New Fairness Metrics

1. Implement metric in `app/core/fairness_metrics.py`
2. Update response schemas in `app/models/schemas.py`
3. Add metric explanations in `app/services/explanation_service.py`

## 🤝 Integration Examples

### Upload and Analyze

```python
import requests

# Upload training data
with open('training.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload/training', 
                           files={'file': f})

# Upload model
with open('model.joblib', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload/model', 
                           files={'file': f})

# Start analysis
analysis_request = {
    "trainingDataPath": "path/to/training.csv",
    "modelPath": "path/to/model.joblib", 
    "targetColumn": "target",
    "autoDetection": True
}
response = requests.post('http://localhost:8000/api/analysis/start', 
                        json=analysis_request)
analysis_id = response.json()['analysisId']

# Check status
status = requests.get(f'http://localhost:8000/api/analysis/status/{analysis_id}')

# Get results
results = requests.get(f'http://localhost:8000/api/analysis/sensitive-features/{analysis_id}')
```

## 📞 Support

- **Documentation**: `/api/docs` for interactive API documentation
- **Health Check**: `/api/health` for service status
- **Logs**: Check application logs for debugging
- **Configuration**: Verify `.env` settings for proper configuration

## 🔄 Version Compatibility

- **Python**: 3.8+
- **FastAPI**: 0.104+
- **Pandas**: 2.1+
- **Scikit-learn**: 1.3+
- **Frontend**: Compatible with React fairness-evaluation-platform

---

**The backend is now ready for production deployment and seamless integration with the React frontend!**
