// API Service Layer for Backend Integration
// This service provides a clean interface between frontend components and backend APIs

// Configuration
const API_CONFIG = {
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api',
  timeout: 30000, // 30 seconds
  retryAttempts: 3,
  retryDelay: 1000, // 1 second
};

// API Client with error handling and retry logic
class APIClient {
  constructor(config = {}) {
    this.baseURL = config.baseURL || API_CONFIG.baseURL;
    this.timeout = config.timeout || API_CONFIG.timeout;
    this.retryAttempts = config.retryAttempts || API_CONFIG.retryAttempts;
    this.retryDelay = config.retryDelay || API_CONFIG.retryDelay;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    let lastError;
    
    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        
        const response = await fetch(url, {
          ...config,
          signal: controller.signal,
        });
        
        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new APIError(
            `HTTP ${response.status}: ${response.statusText}`,
            response.status,
            await response.text()
          );
        }

        return await response.json();
      } catch (error) {
        lastError = error;
        
        // Don't retry on client errors (4xx) or abort errors
        if (error.status >= 400 && error.status < 500) {
          throw error;
        }
        
        if (error.name === 'AbortError') {
          throw new APIError('Request timeout', 408, 'Request timed out');
        }

        // Wait before retrying (exponential backoff)
        if (attempt < this.retryAttempts) {
          await new Promise(resolve => 
            setTimeout(resolve, this.retryDelay * Math.pow(2, attempt - 1))
          );
        }
      }
    }

    throw lastError;
  }

  get(endpoint, params = {}) {
    const url = new URL(endpoint, this.baseURL);
    Object.keys(params).forEach(key => {
      if (params[key] !== undefined && params[key] !== null) {
        url.searchParams.append(key, params[key]);
      }
    });
    
    return this.request(url.pathname + url.search, {
      method: 'GET',
    });
  }

  post(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  put(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  delete(endpoint) {
    return this.request(endpoint, {
      method: 'DELETE',
    });
  }

  // File upload with FormData
  async uploadFile(endpoint, file, additionalData = {}) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Add additional form fields
    Object.keys(additionalData).forEach(key => {
      formData.append(key, additionalData[key]);
    });

    return this.request(endpoint, {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
    });
  }
}

// Custom API Error class
class APIError extends Error {
  constructor(message, status, response) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.response = response;
  }
}

// Create API client instance
const apiClient = new APIClient();

// API Service Methods
export const APIService = {
  // Dataset Upload APIs
  async uploadTrainingData(file, metadata = {}) {
    try {
      return await apiClient.uploadFile('/upload/training', file, metadata);
    } catch (error) {
      console.error('Training data upload failed:', error);
      throw new APIError('Failed to upload training data', error.status, error.response);
    }
  },

  async uploadTestingData(file, metadata = {}) {
    try {
      return await apiClient.uploadFile('/upload/testing', file, metadata);
    } catch (error) {
      console.error('Testing data upload failed:', error);
      throw new APIError('Failed to upload testing data', error.status, error.response);
    }
  },

  async uploadModel(file, metadata = {}) {
    try {
      return await apiClient.uploadFile('/upload/model', file, metadata);
    } catch (error) {
      console.error('Model upload failed:', error);
      throw new APIError('Failed to upload model', error.status, error.response);
    }
  },

  // Analysis Management APIs
  async startAnalysis(config) {
    try {
      return await apiClient.post('/analysis/start', config);
    } catch (error) {
      console.error('Analysis start failed:', error);
      throw new APIError('Failed to start analysis', error.status, error.response);
    }
  },

  async getAnalysisStatus(analysisId) {
    try {
      return await apiClient.get(`/analysis/status/${analysisId}`);
    } catch (error) {
      console.error('Analysis status check failed:', error);
      throw new APIError('Failed to get analysis status', error.status, error.response);
    }
  },

  async stopAnalysis(analysisId) {
    try {
      return await apiClient.delete(`/analysis/${analysisId}`);
    } catch (error) {
      console.error('Analysis stop failed:', error);
      throw new APIError('Failed to stop analysis', error.status, error.response);
    }
  },

  // Results Retrieval APIs
  async getSensitiveFeatures(analysisId) {
    try {
      return await apiClient.get(`/analysis/sensitive-features/${analysisId}`);
    } catch (error) {
      console.error('Sensitive features retrieval failed:', error);
      throw new APIError('Failed to get sensitive features', error.status, error.response);
    }
  },

  async getFairnessMetrics(analysisId, attribute = null) {
    try {
      const params = attribute ? { attribute } : {};
      return await apiClient.get(`/analysis/fairness-metrics/${analysisId}`, params);
    } catch (error) {
      console.error('Fairness metrics retrieval failed:', error);
      throw new APIError('Failed to get fairness metrics', error.status, error.response);
    }
  },

  async getMitigationStrategies(analysisId) {
    try {
      return await apiClient.get(`/analysis/mitigation-strategies/${analysisId}`);
    } catch (error) {
      console.error('Mitigation strategies retrieval failed:', error);
      throw new APIError('Failed to get mitigation strategies', error.status, error.response);
    }
  },

  async getBeforeAfterComparison(analysisId, strategy) {
    try {
      return await apiClient.get(`/analysis/before-after-comparison/${analysisId}`, { strategy });
    } catch (error) {
      console.error('Before/after comparison retrieval failed:', error);
      throw new APIError('Failed to get before/after comparison', error.status, error.response);
    }
  },

  // Configuration APIs
  async getAvailableSensitiveAttributes() {
    try {
      return await apiClient.get('/config/sensitive-attributes');
    } catch (error) {
      console.error('Sensitive attributes retrieval failed:', error);
      throw new APIError('Failed to get sensitive attributes', error.status, error.response);
    }
  },

  async getMitigationOptions() {
    try {
      return await apiClient.get('/config/mitigation-options');
    } catch (error) {
      console.error('Mitigation options retrieval failed:', error);
      throw new APIError('Failed to get mitigation options', error.status, error.response);
    }
  },

  // Health Check
  async healthCheck() {
    try {
      return await apiClient.get('/health');
    } catch (error) {
      console.error('Health check failed:', error);
      throw new APIError('Health check failed', error.status, error.response);
    }
  },
};

// Development Mode Detection
export const isDevelopmentMode = () => {
  return process.env.NODE_ENV === 'development' || 
         process.env.REACT_APP_USE_MOCK_DATA === 'true';
};

// API Service with Mock Fallback
export const createAPIServiceWithMockFallback = (mockService) => {
  const serviceProxy = new Proxy(APIService, {
    get(target, prop) {
      if (isDevelopmentMode() && mockService && typeof mockService[prop] === 'function') {
        return mockService[prop].bind(mockService);
      }
      return target[prop];
    }
  });

  return serviceProxy;
};

// Export APIError for error handling
export { APIError, APIClient };

// Export default configured API service
export default APIService;
