// Enhanced Mock Service for Development
// This service implements the same interface as APIService but returns mock data
// It simulates real API behavior including delays, loading states, and error scenarios

import {
  mockSensitiveFeatures,
  mockFairnessMetricsByAttribute,
  mockMitigationStrategies,
  mockBeforeAfterComparison,
  SENSITIVE_ATTRIBUTES,
} from './mockDataService';

// Simulate network delay
const simulateDelay = (min = 500, max = 1500) => {
  const delay = Math.random() * (max - min) + min;
  return new Promise(resolve => setTimeout(resolve, delay));
};

// Simulate API errors occasionally
const simulateError = (errorRate = 0.05) => {
  if (Math.random() < errorRate) {
    throw new Error('Simulated network error');
  }
};

// Mock file upload response
const createMockUploadResponse = (file, type) => ({
  success: true,
  uploadId: `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
  filename: file.name,
  size: file.size,
  type: file.type,
  uploadType: type,
  timestamp: new Date().toISOString(),
  metadata: {
    columns: type === 'training' || type === 'testing' ? [
      'age', 'gender', 'race', 'education', 'income', 
      'employment_status', 'credit_score', 'target'
    ] : null,
    rows: type === 'training' ? 10000 : type === 'testing' ? 2500 : null,
    validation: {
      isValid: true,
      errors: [],
      warnings: []
    }
  }
});

// Mock analysis response
const createMockAnalysisResponse = () => ({
  analysisId: `analysis_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
  status: 'started',
  createdAt: new Date().toISOString(),
  estimatedDuration: 120, // seconds
  steps: [
    'Data validation',
    'Sensitive feature detection', 
    'Fairness metric calculation',
    'Mitigation strategy analysis'
  ]
});

// Enhanced Mock Service
export const EnhancedMockService = {
  // Dataset Upload APIs
  async uploadTrainingData(file, metadata = {}) {
    await simulateDelay(1000, 3000); // Longer for file uploads
    simulateError(0.02); // 2% error rate
    
    console.log('Mock: Uploading training data:', file.name);
    return createMockUploadResponse(file, 'training');
  },

  async uploadTestingData(file, metadata = {}) {
    await simulateDelay(1000, 3000);
    simulateError(0.02);
    
    console.log('Mock: Uploading testing data:', file.name);
    return createMockUploadResponse(file, 'testing');
  },

  async uploadModel(file, metadata = {}) {
    await simulateDelay(800, 2000);
    simulateError(0.02);
    
    console.log('Mock: Uploading model:', file.name);
    return createMockUploadResponse(file, 'model');
  },

  // Analysis Management APIs
  async startAnalysis(config) {
    await simulateDelay(500, 1000);
    simulateError(0.03);
    
    console.log('Mock: Starting analysis with config:', config);
    const response = createMockAnalysisResponse();
    
    // Store analysis in mock state for status tracking
    const analysisState = {
      id: response.analysisId,
      status: 'running',
      progress: 0,
      currentStep: 'Data validation',
      startTime: Date.now(),
    };
    
    // Simulate progress updates
    this._simulateAnalysisProgress(analysisState);
    
    return response;
  },

  async getAnalysisStatus(analysisId) {
    await simulateDelay(200, 500);
    simulateError(0.01);
    
    // Simulate different analysis states
    const elapsed = Date.now() - (this._analysisStartTime || Date.now());
    let status, progress, currentStep;
    
    if (elapsed < 30000) { // First 30 seconds
      status = 'running';
      progress = Math.min(25, (elapsed / 30000) * 25);
      currentStep = 'Data validation';
    } else if (elapsed < 60000) { // Next 30 seconds
      status = 'running';
      progress = 25 + Math.min(25, ((elapsed - 30000) / 30000) * 25);
      currentStep = 'Sensitive feature detection';
    } else if (elapsed < 90000) { // Next 30 seconds
      status = 'running';
      progress = 50 + Math.min(30, ((elapsed - 60000) / 30000) * 30);
      currentStep = 'Fairness metric calculation';
    } else if (elapsed < 120000) { // Final 30 seconds
      status = 'running';
      progress = 80 + Math.min(20, ((elapsed - 90000) / 30000) * 20);
      currentStep = 'Mitigation strategy analysis';
    } else {
      status = 'completed';
      progress = 100;
      currentStep = 'Analysis complete';
    }

    return {
      analysisId,
      status,
      progress,
      currentStep,
      estimatedTimeRemaining: status === 'completed' ? 0 : Math.max(0, 120 - elapsed / 1000),
      results: status === 'completed' ? {
        sensitiveFeatures: true,
        fairnessMetrics: true,
        mitigationStrategies: true,
        beforeAfterComparisons: true
      } : null
    };
  },

  async stopAnalysis(analysisId) {
    await simulateDelay(300, 800);
    simulateError(0.02);
    
    console.log('Mock: Stopping analysis:', analysisId);
    return {
      analysisId,
      status: 'stopped',
      message: 'Analysis stopped successfully'
    };
  },

  // Results Retrieval APIs
  async getSensitiveFeatures(analysisId) {
    await simulateDelay(400, 800);
    simulateError(0.02);
    
    console.log('Mock: Getting sensitive features for analysis:', analysisId);
    return {
      analysisId,
      ...mockSensitiveFeatures
    };
  },

  async getFairnessMetrics(analysisId, attribute = null) {
    await simulateDelay(600, 1200);
    simulateError(0.02);
    
    console.log('Mock: Getting fairness metrics for analysis:', analysisId, 'attribute:', attribute);
    
    if (attribute) {
      return {
        analysisId,
        attribute,
        data: mockFairnessMetricsByAttribute[attribute] || mockFairnessMetricsByAttribute.gender
      };
    }
    
    return {
      analysisId,
      data: mockFairnessMetricsByAttribute
    };
  },

  async getMitigationStrategies(analysisId) {
    await simulateDelay(500, 1000);
    simulateError(0.02);
    
    console.log('Mock: Getting mitigation strategies for analysis:', analysisId);
    return {
      analysisId,
      ...mockMitigationStrategies
    };
  },

  async getBeforeAfterComparison(analysisId, strategy) {
    await simulateDelay(800, 1500);
    simulateError(0.02);
    
    console.log('Mock: Getting before/after comparison for analysis:', analysisId, 'strategy:', strategy);
    
    // Return data for the first available attribute if specific strategy comparison exists
    const attributeKey = Object.keys(mockBeforeAfterComparison)[0];
    return {
      analysisId,
      strategy,
      data: mockBeforeAfterComparison[attributeKey] || mockBeforeAfterComparison.gender
    };
  },

  // Configuration APIs
  async getAvailableSensitiveAttributes() {
    await simulateDelay(200, 400);
    simulateError(0.01);
    
    return {
      attributes: Object.values(SENSITIVE_ATTRIBUTES).map(attr => ({
        name: attr,
        displayName: attr.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
        description: `${attr.replace('_', ' ')} sensitive attribute`,
        dataType: 'categorical',
        required: attr === 'gender' // Make gender required for demo
      }))
    };
  },

  async getMitigationOptions() {
    await simulateDelay(300, 600);
    simulateError(0.01);
    
    return {
      strategies: mockMitigationStrategies.strategies.map(strategy => ({
        name: strategy.name,
        category: strategy.category,
        description: strategy.description,
        supportedAttributes: strategy.targetAttributes,
        complexity: strategy.stars <= 1 ? 'low' : strategy.stars === 2 ? 'medium' : 'high'
      }))
    };
  },

  // Health Check
  async healthCheck() {
    await simulateDelay(100, 300);
    
    return {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: '1.0.0-mock',
      environment: 'development',
      services: {
        database: 'healthy',
        mlEngine: 'healthy',
        fileStorage: 'healthy'
      }
    };
  },

  // Internal helper methods
  _simulateAnalysisProgress(analysisState) {
    this._analysisStartTime = analysisState.startTime;
    
    // Simulate periodic status updates
    const updateInterval = setInterval(() => {
      const elapsed = Date.now() - analysisState.startTime;
      
      if (elapsed >= 120000) { // Complete after 2 minutes
        clearInterval(updateInterval);
        analysisState.status = 'completed';
        analysisState.progress = 100;
        console.log('Mock: Analysis completed');
      }
    }, 5000);
  }
};

export default EnhancedMockService;
