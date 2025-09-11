// Data Contracts for Backend Integration
// These interfaces define the expected data structure between frontend and backend

/**
 * SENSITIVE FEATURE DETECTION RESPONSE
 * Expected from: GET /api/analysis/sensitive-features/{analysis_id}
 */
export const SensitiveFeatureDetectionContract = {
  detectedFeatures: [
    {
      name: 'string',              // e.g., 'gender', 'age_group', 'race'
      dataType: 'string',          // 'Categorical', 'Numerical', 'Binary'
      test: 'string',              // 'Chi-Square', 'ANOVA', 'T-Test'
      pValue: 'number',            // 0.001
      effectSize: 'number',        // 0.31
      effectSizeLabel: 'string',   // 'Large', 'Medium', 'Small'
      correlation: 'number',       // 0.28
      sensitivityLevel: 'string',  // 'Highly Sensitive', 'Moderately Sensitive'
      groups: ['string'],          // ['Male', 'Female'] or ['Young', 'Middle', 'Senior']
      description: 'string'        // Human-readable description
    }
  ],
  summary: {
    totalDetected: 'number',           // 3
    highlySensitiveCount: 'number',    // 2
    moderatelySensitiveCount: 'number', // 1
    riskLevel: 'string'                // 'HIGH', 'MEDIUM', 'LOW'
  }
};

/**
 * FAIRNESS METRICS BY ATTRIBUTE RESPONSE
 * Expected from: GET /api/analysis/fairness-metrics/{analysis_id}?attribute={attribute_name}
 * Or: GET /api/analysis/fairness-metrics/{analysis_id} (returns all attributes)
 */
export const FairnessMetricsContract = {
  // Key: attribute name (e.g., 'gender', 'age_group', 'race')
  [attributeName]: {
    attribute: 'string',        // 'gender'
    groups: ['string'],         // ['Male', 'Female']
    metrics: [
      {
        name: 'string',                    // 'Statistical Parity'
        value: 'number',                   // 0.23
        status: 'string',                  // 'Biased', 'Fair', 'Warning'
        threshold: 'number',               // 0.10
        groupRates: {                      // Key-value pairs for each group
          [groupName]: 'number'            // 'Male': 0.65, 'Female': 0.42
        },
        description: 'string',             // Human-readable description
        tooltip: 'string'                  // Additional context for UI
      }
    ],
    overallScore: 'number',              // 42 (0-100)
    overallRating: 'string',             // 'Poor Fairness', 'Moderate Fairness', 'Good Fairness'
    riskClassification: 'string',        // 'High Bias Risk', 'Medium Bias Risk', 'Low Bias Risk'
    primaryIssues: ['string']            // ['Gender discrimination', 'Age bias']
  }
};

/**
 * MITIGATION STRATEGIES RESPONSE
 * Expected from: GET /api/analysis/mitigation-strategies/{analysis_id}
 */
export const MitigationStrategiesContract = {
  strategies: [
    {
      name: 'string',                    // 'Reweighing'
      category: 'string',                // 'Preprocessing', 'In-processing', 'Post-processing'
      fairnessImprovement: 'number',     // 38 (percentage)
      accuracyImpact: 'number',          // -1.8 (percentage, negative means decrease)
      precisionImpact: 'number',         // -1.2
      f1Impact: 'number',                // -1.5
      stars: 'number',                   // 1-3 rating
      recommendation: 'string',          // 'Highly Recommended', 'Recommended', 'Good option'
      description: 'string',             // Human-readable description
      targetAttributes: ['string']       // ['gender', 'age_group', 'race'] - which attributes this strategy addresses
    }
  ]
};

/**
 * BEFORE/AFTER COMPARISON RESPONSE
 * Expected from: GET /api/analysis/before-after-comparison/{analysis_id}?strategy={strategy_name}
 */
export const BeforeAfterComparisonContract = {
  strategy: 'string',                    // 'Reweighing'
  
  // Fairness metrics comparison
  fairnessMetrics: {
    before: ['number'],                  // [42, 38, 45, 35, 48, 41] - scores for each metric
    after: ['number'],                   // [78, 72, 81, 69, 75, 77]
    metrics: ['string']                  // ['Statistical Parity', 'Disparate Impact', ...]
  },
  
  // Performance metrics comparison
  performance: {
    before: {
      accuracy: 'number',                // 85.2
      precision: 'number',               // 82.1
      recall: 'number',                  // 79.8
      f1: 'number'                       // 80.9
    },
    after: {
      accuracy: 'number',                // 83.4
      precision: 'number',               // 80.9
      recall: 'number',                  // 78.6
      f1: 'number'                       // 79.4
    }
  },
  
  // Group-wise comparison by sensitive attribute
  groupComparisons: {
    // Key: attribute name
    [attributeName]: {
      attribute: 'string',               // 'gender'
      groups: ['string'],                // ['Male', 'Female']
      before: {
        [groupName]: 'number'            // 'Male': 65, 'Female': 42
      },
      after: {
        [groupName]: 'number'            // 'Male': 58, 'Female': 55
      },
      improvement: {
        [groupName]: 'number'            // 'Male': -7, 'Female': 13
      }
    }
  }
};

/**
 * UPLOAD RESPONSE CONTRACTS
 * Expected from: POST /api/upload/training, /api/upload/testing, /api/upload/model
 */
export const UploadResponseContract = {
  success: 'boolean',                    // true
  filename: 'string',                    // 'training_data.csv'
  size: 'number',                        // 1024000 (bytes)
  columns: ['string'],                   // ['age', 'gender', 'income', 'target'] (for CSV files)
  preview: {
    rows: 'number',                      // 1000
    sampleData: [{}]                     // Array of sample rows
  },
  validation: {
    isValid: 'boolean',                  // true
    errors: ['string'],                  // ['Missing values in column X']
    warnings: ['string']                 // ['High correlation between X and Y']
  }
};

/**
 * ANALYSIS START RESPONSE
 * Expected from: POST /api/analysis/start
 */
export const AnalysisStartContract = {
  analysisId: 'string',                  // 'uuid-string'
  status: 'string',                      // 'started', 'running', 'completed', 'failed'
  estimatedDuration: 'number',           // 120 (seconds)
  steps: [
    {
      name: 'string',                    // 'sensitive_feature_detection'
      status: 'string',                  // 'pending', 'running', 'completed'
      progress: 'number'                 // 0-100
    }
  ]
};

/**
 * FRONTEND STATE STRUCTURE
 * How the frontend should structure its state to be compatible with backend responses
 */
export const FrontendStateContract = {
  datasets: {
    training: 'UploadResponseContract | null',
    testing: 'UploadResponseContract | null',
    model: 'UploadResponseContract | null'
  },
  configuration: {
    targetColumn: 'string | null',
    sensitiveAttributes: ['string'],
    autoDetectionEnabled: 'boolean'
  },
  analysis: {
    id: 'string | null',
    isRunning: 'boolean',
    isComplete: 'boolean',
    results: {
      sensitiveFeatures: 'SensitiveFeatureDetectionContract | null',
      fairnessMetrics: 'FairnessMetricsContract | null',
      mitigationStrategies: 'MitigationStrategiesContract | null',
      beforeAfterComparisons: 'BeforeAfterComparisonContract | null'
    }
  },
  ui: {
    selectedSensitiveAttribute: 'string',     // 'gender'
    selectedMitigationStrategy: 'string',     // 'Reweighing'
    activeTab: 'number',                      // 0, 1, 2, 3
    activeSection: 'string'                   // 'detection', 'metrics', 'mitigation'
  }
};

/**
 * API ENDPOINT MAPPINGS
 * Maps frontend actions to backend endpoints
 */
export const APIEndpoints = {
  // File uploads
  uploadTraining: 'POST /api/upload/training',
  uploadTesting: 'POST /api/upload/testing', 
  uploadModel: 'POST /api/upload/model',
  
  // Analysis lifecycle
  startAnalysis: 'POST /api/analysis/start',
  getAnalysisStatus: 'GET /api/analysis/status/{analysis_id}',
  
  // Results retrieval
  getSensitiveFeatures: 'GET /api/analysis/sensitive-features/{analysis_id}',
  getFairnessMetrics: 'GET /api/analysis/fairness-metrics/{analysis_id}',
  getFairnessMetricsByAttribute: 'GET /api/analysis/fairness-metrics/{analysis_id}?attribute={attribute}',
  getMitigationStrategies: 'GET /api/analysis/mitigation-strategies/{analysis_id}',
  getBeforeAfterComparison: 'GET /api/analysis/before-after-comparison/{analysis_id}?strategy={strategy}',
  
  // AI explanations
  getMetricExplanation: 'POST /api/explanations/metric',
  getDetectionExplanation: 'POST /api/explanations/detection',
  getMitigationExplanation: 'POST /api/explanations/mitigation'
};

/**
 * UTILITY FUNCTIONS FOR DATA TRANSFORMATION
 */
export const DataTransformers = {
  // Transform backend response to frontend format
  transformSensitiveFeatures: (backendResponse) => {
    // Add any transformation logic here
    return backendResponse;
  },
  
  // Transform frontend request to backend format
  transformAnalysisRequest: (frontendState) => {
    return {
      trainingDataPath: frontendState.datasets.training?.path,
      testingDataPath: frontendState.datasets.testing?.path,
      modelPath: frontendState.datasets.model?.path,
      targetColumn: frontendState.configuration.targetColumn,
      sensitiveAttributes: frontendState.configuration.sensitiveAttributes,
      autoDetection: frontendState.configuration.autoDetectionEnabled
    };
  },
  
  // Validate response structure
  validateResponse: (response, expectedContract) => {
    // Add validation logic here
    return true;
  }
};

export default {
  SensitiveFeatureDetectionContract,
  FairnessMetricsContract,
  MitigationStrategiesContract,
  BeforeAfterComparisonContract,
  UploadResponseContract,
  AnalysisStartContract,
  FrontendStateContract,
  APIEndpoints,
  DataTransformers
};
