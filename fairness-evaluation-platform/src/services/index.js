// Service Factory for API Integration
// Automatically switches between real API service and mock service based on environment

import APIService from './apiService';
import EnhancedMockService from './enhancedMockService';

// Check if we should use mock data
const shouldUseMockData = () => {
  // Use mock data when explicitly enabled, but not automatically in development
  return process.env.REACT_APP_USE_MOCK_DATA === 'true';
};

// Check if backend is available
const checkBackendAvailability = async () => {
  try {
    await APIService.healthCheck();
    return true;
  } catch (error) {
    console.warn('Backend not available, falling back to mock service:', error.message);
    return false;
  }
};

// Service Factory Class
class ServiceFactory {
  constructor() {
    this.activeService = null;
    this.isInitialized = false;
    this.initializationPromise = null;
  }

  async initialize() {
    if (this.isInitialized) {
      return this.activeService;
    }

    if (this.initializationPromise) {
      return this.initializationPromise;
    }

    this.initializationPromise = this._performInitialization();
    return this.initializationPromise;
  }

  async _performInitialization() {
    console.log('Initializing service factory...');

    // Force mock service in development or when explicitly set
    if (shouldUseMockData()) {
      console.log('Using mock service (development mode)');
      this.activeService = EnhancedMockService;
      this.isInitialized = true;
      return this.activeService;
    }

    // Try to connect to real backend
    console.log('Attempting to connect to backend API...');
    const backendAvailable = await checkBackendAvailability();

    if (backendAvailable) {
      console.log('Using real API service');
      this.activeService = APIService;
    } else {
      console.log('Using mock service (backend unavailable)');
      this.activeService = EnhancedMockService;
    }

    this.isInitialized = true;
    return this.activeService;
  }

  // Get the active service (with automatic initialization)
  async getService() {
    return await this.initialize();
  }

  // Force switch to mock service (useful for testing)
  useMockService() {
    console.log('Forcing mock service usage');
    this.activeService = EnhancedMockService;
    this.isInitialized = true;
  }

  // Force switch to real API service
  useRealService() {
    console.log('Forcing real API service usage');
    this.activeService = APIService;
    this.isInitialized = true;
  }

  // Reset the factory (useful for testing)
  reset() {
    this.activeService = null;
    this.isInitialized = false;
    this.initializationPromise = null;
  }

  // Check which service is currently active
  getActiveServiceType() {
    if (!this.isInitialized) return 'uninitialized';
    return this.activeService === APIService ? 'real' : 'mock';
  }
}

// Create singleton instance
const serviceFactory = new ServiceFactory();

// Exported service interface
export const FairnessAPIService = {
  // Initialize and get service
  async initialize() {
    return await serviceFactory.initialize();
  },

  // Dataset Upload APIs
  async uploadTrainingData(file, metadata = {}) {
    const service = await serviceFactory.getService();
    return service.uploadTrainingData(file, metadata);
  },

  async uploadTestingData(file, metadata = {}) {
    const service = await serviceFactory.getService();
    return service.uploadTestingData(file, metadata);
  },

  async uploadModel(file, metadata = {}) {
    const service = await serviceFactory.getService();
    return service.uploadModel(file, metadata);
  },

  // Analysis Management APIs
  async startAnalysis(config) {
    const service = await serviceFactory.getService();
    return service.startAnalysis(config);
  },

  async getAnalysisStatus(analysisId) {
    const service = await serviceFactory.getService();
    return service.getAnalysisStatus(analysisId);
  },

  async stopAnalysis(analysisId) {
    const service = await serviceFactory.getService();
    return service.stopAnalysis(analysisId);
  },

  // Results Retrieval APIs
  async getSensitiveFeatures(analysisId) {
    const service = await serviceFactory.getService();
    return service.getSensitiveFeatures(analysisId);
  },

  async getFairnessMetrics(analysisId, attribute = null) {
    const service = await serviceFactory.getService();
    return service.getFairnessMetrics(analysisId, attribute);
  },

  async getMitigationStrategies(analysisId) {
    const service = await serviceFactory.getService();
    return service.getMitigationStrategies(analysisId);
  },

  async getBeforeAfterComparison(analysisId, strategy) {
    const service = await serviceFactory.getService();
    return service.getBeforeAfterComparison(analysisId, strategy);
  },

  // Real Mitigation Application
  async applyRealMitigation(analysisId, strategyName) {
    const service = await serviceFactory.getService();
    return service.applyRealMitigation(analysisId, strategyName);
  },

  // Configuration APIs
  async getAvailableSensitiveAttributes() {
    const service = await serviceFactory.getService();
    return service.getAvailableSensitiveAttributes();
  },

  async getMitigationOptions() {
    const service = await serviceFactory.getService();
    return service.getMitigationOptions();
  },

  // Health Check
  async healthCheck() {
    const service = await serviceFactory.getService();
    return service.healthCheck();
  },

  // Utility methods
  getActiveServiceType() {
    return serviceFactory.getActiveServiceType();
  },

  isUsingMockService() {
    return serviceFactory.getActiveServiceType() === 'mock';
  },

  isUsingRealService() {
    return serviceFactory.getActiveServiceType() === 'real';
  },

  // Force service type (for testing/debugging)
  forceMockService() {
    serviceFactory.useMockService();
  },

  forceRealService() {
    serviceFactory.useRealService();
  },

  reset() {
    serviceFactory.reset();
  }
};

export default FairnessAPIService;
