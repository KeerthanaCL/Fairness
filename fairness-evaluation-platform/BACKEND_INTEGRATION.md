# Fairness Evaluation Platform - Backend Integration Guide

## 🚀 Backend Integration Readiness

This frontend application is **fully prepared for backend API integration** with a comprehensive service layer, data contracts, and development tools.

## 📁 Project Structure

```
src/
├── services/
│   ├── apiService.js              # Real API service implementation
│   ├── enhancedMockService.js     # Development mock service
│   ├── mockDataService.js         # Mock data provider
│   └── index.js                   # Service factory (auto-switches)
├── hooks/
│   └── useAPI.js                  # Custom hooks for API integration
├── contracts/
│   └── dataContracts.js           # API data contracts & endpoints
├── context/
│   └── AppContext.js              # Global state management
└── components/                    # UI components (API-ready)
```

## 🔧 Backend Integration Features

### ✅ Complete API Service Layer
- **Real API Service** (`apiService.js`): Production-ready HTTP client with retry logic
- **Mock Service** (`enhancedMockService.js`): Development simulation with realistic delays
- **Service Factory** (`index.js`): Automatic backend detection and fallback
- **Custom Hooks** (`useAPI.js`): React hooks for data fetching, file uploads, and polling

### ✅ Comprehensive Data Contracts
- **Defined API Endpoints**: Complete URL mappings for all backend services
- **TypeScript-like Schemas**: Clear data structure contracts for all API responses
- **Error Handling**: Standardized error response formats
- **File Upload Contracts**: Multi-part form data specifications

### ✅ Environment Configuration
- **Development**: Mock data enabled, local backend URLs
- **Production**: Real API endpoints, optimized settings
- **Configurable**: All endpoints and settings via environment variables

### ✅ Error Handling & UX
- **Loading States**: Spinners and progress indicators
- **Error Recovery**: Retry mechanisms and user-friendly error messages
- **Offline Fallback**: Graceful degradation to mock data
- **Real-time Updates**: Polling for analysis status updates

## 🌐 API Endpoints

### File Upload APIs
```
POST /api/upload/training     - Upload training dataset
POST /api/upload/testing      - Upload testing dataset  
POST /api/upload/model        - Upload ML model file
```

### Analysis Management APIs
```
POST   /api/analysis/start                           - Start fairness analysis
GET    /api/analysis/status/{analysis_id}           - Get analysis status
DELETE /api/analysis/{analysis_id}                  - Stop analysis
```

### Results Retrieval APIs
```
GET /api/analysis/sensitive-features/{analysis_id}                    - Get detected sensitive features
GET /api/analysis/fairness-metrics/{analysis_id}?attribute={attr}     - Get fairness metrics
GET /api/analysis/mitigation-strategies/{analysis_id}                 - Get mitigation recommendations
GET /api/analysis/before-after-comparison/{analysis_id}?strategy={s}  - Get comparison results
```

### Configuration APIs
```
GET /api/config/sensitive-attributes  - Get available sensitive attributes
GET /api/config/mitigation-options    - Get mitigation strategy options
GET /api/health                       - Health check endpoint
```

## 🛠️ Development Setup

### Environment Variables

Create `.env.development`:
```bash
REACT_APP_API_BASE_URL=http://localhost:8000/api
REACT_APP_USE_MOCK_DATA=true
REACT_APP_DEBUG_MODE=true
```

Create `.env.production`:
```bash
REACT_APP_API_BASE_URL=https://your-backend-api.com/api
REACT_APP_USE_MOCK_DATA=false
REACT_APP_DEBUG_MODE=false
```

### Service Usage Example

```javascript
import FairnessAPIService from '../services';

// The service automatically chooses between real API and mock data
const component = () => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Automatically uses real API if available, mock data otherwise
        const result = await FairnessAPIService.getSensitiveFeatures(analysisId);
        setData(result);
      } catch (error) {
        console.error('API call failed:', error);
      }
    };
    
    fetchData();
  }, [analysisId]);
};
```

### Using Custom Hooks

```javascript
import { useFairnessMetrics, useFileUpload, useAnalysis } from '../hooks/useAPI';

const MyComponent = () => {
  // Automatic data fetching with loading/error states
  const { data, loading, error, refresh } = useFairnessMetrics(analysisId, 'gender');
  
  // File upload with progress tracking
  const { uploadFile, uploads } = useFileUpload();
  
  // Analysis management with polling
  const { startAnalysis, analysis, isRunning } = useAnalysis();
  
  return (
    <div>
      {loading && <Spinner />}
      {error && <ErrorMessage error={error} onRetry={refresh} />}
      {data && <DataVisualization data={data} />}
    </div>
  );
};
```

## 🔄 Backend Integration Steps

### 1. **Update Environment Variables**
```bash
# Point to your backend API
REACT_APP_API_BASE_URL=http://your-backend-url:8000/api
REACT_APP_USE_MOCK_DATA=false
```

### 2. **Implement Backend Endpoints**
The frontend expects these exact endpoint contracts (see `contracts/dataContracts.js`):

```python
# Example Flask/FastAPI endpoints to implement

@app.post("/api/upload/training")
async def upload_training_data(file: UploadFile):
    # Return: { uploadId, filename, size, metadata, validation }

@app.post("/api/analysis/start") 
async def start_analysis(config: AnalysisConfig):
    # Return: { analysisId, status, estimatedDuration, steps }

@app.get("/api/analysis/fairness-metrics/{analysis_id}")
async def get_fairness_metrics(analysis_id: str, attribute: str = None):
    # Return: Contract defined in dataContracts.js
```

### 3. **Test Integration**
```bash
# Test with real backend
npm start

# Check browser console for API calls
# Service factory will auto-detect backend availability
```

### 4. **Error Handling**
The frontend handles these scenarios automatically:
- ✅ Backend unavailable → Falls back to mock data
- ✅ Network errors → Retry with exponential backoff  
- ✅ API errors → User-friendly error messages
- ✅ Timeout → Configurable timeout handling

## 📊 Mock Data vs Real API

### Development Mode (Mock Data)
- ✅ Realistic response delays (500ms-2s)
- ✅ Complete multi-attribute support (Gender, Age, Race)
- ✅ Simulated analysis progress tracking
- ✅ Error simulation for testing edge cases
- ✅ File upload simulation with progress

### Production Mode (Real API)
- ✅ HTTP client with timeout and retry logic
- ✅ Automatic error recovery mechanisms
- ✅ File upload with FormData support
- ✅ Real-time analysis polling
- ✅ Backend health monitoring

## 🚨 Components Ready for Backend

All components are designed to work with dynamic data:

### ✅ File Upload Components
- `FileUploader.js` - Handles file validation and upload
- Supports training data, testing data, and model files
- Progress tracking and error handling

### ✅ Analysis Components  
- `SensitiveFeatureDetection.js` - Displays detected sensitive features
- `FairnessMetricsDashboard.js` - Shows fairness metrics with attribute switching
- `MitigationStrategyAnalysis.js` - Recommends mitigation strategies

### ✅ Visualization Components
- `FairnessRadarChart.js` - Dynamic radar charts for fairness metrics
- `ComparisonBarChart.js` - Before/after comparison visualizations
- All charts support dynamic data updates

### ✅ Context & State Management
- `AppContext.js` - Global state for datasets, analysis results, UI state
- Reducer pattern for predictable state updates
- Context providers for easy data sharing

## 🔐 Security Considerations

### API Security
- CORS configuration needed for frontend domain
- Authentication tokens (if required) can be added to `apiService.js`
- File upload validation on both frontend and backend
- API rate limiting recommendations

### File Upload Security
- File type validation (CSV, JSON, PKL, JOBLIB)
- File size limits (configurable via environment)
- Virus scanning integration points available
- Secure file storage recommendations

## 📈 Performance Optimizations

### Frontend Optimizations
- ✅ API response caching (5-10 minute TTL)
- ✅ Debounced API calls for user interactions
- ✅ Lazy loading for large datasets
- ✅ Memoized expensive computations

### Backend Integration Optimizations  
- ✅ Request/response compression support
- ✅ Pagination support for large result sets
- ✅ WebSocket ready for real-time updates
- ✅ Batch API calls for efficiency

## 🧪 Testing Strategy

### API Integration Testing
```bash
# Test with mock backend
REACT_APP_USE_MOCK_DATA=true npm start

# Test with real backend  
REACT_APP_USE_MOCK_DATA=false npm start

# Test error scenarios
# Mock service includes 2-5% error simulation
```

### Component Testing
- All components accept `analysisId` prop for dynamic data
- Components gracefully handle loading, error, and empty states
- Mock data provides comprehensive test scenarios

## 📝 Next Steps for Backend Development

1. **Implement Core Endpoints**: Start with file upload and analysis start endpoints
2. **Add Authentication**: Integrate user management and API keys
3. **Database Integration**: Store analysis results and user data
4. **ML Pipeline**: Connect to actual fairness analysis algorithms
5. **Real-time Updates**: WebSocket or SSE for live analysis progress
6. **Monitoring**: Add logging, metrics, and health checks

## 🎯 Summary

**The frontend is 100% ready for backend integration** with:

- ✅ **Complete API Service Layer** - Production-ready HTTP client
- ✅ **Comprehensive Data Contracts** - All endpoint schemas defined  
- ✅ **Automatic Fallback System** - Mock data when backend unavailable
- ✅ **Error Handling & Recovery** - Retry logic and user feedback
- ✅ **Environment Configuration** - Easy deployment configuration
- ✅ **React Hooks Integration** - Clean component integration patterns
- ✅ **Loading & Error States** - Professional UX for all scenarios
- ✅ **File Upload System** - Multi-part uploads with progress tracking
- ✅ **Real-time Polling** - Analysis progress monitoring
- ✅ **Dynamic Components** - All UI components support live data

**Simply update the environment variables to point to your backend API and the entire application will seamlessly switch from mock data to real API calls.**
