// Mock Data Service for Multi-Sensitive Feature Support
// This service simulates backend responses with dynamic sensitive attribute support

export const SENSITIVE_ATTRIBUTES = {
  GENDER: 'gender',
  AGE_GROUP: 'age_group',
  AGE: 'age', 
  RACE: 'race',
  EDUCATION_YEARS: 'education_years',
  WORK_EXPERIENCE: 'work_experience',
  INCOME: 'income',
  MARITAL_STATUS: 'marital_status'
};

// Mock Sensitive Feature Detection Results
export const mockSensitiveFeatures = {
  detectedFeatures: [
    {
      name: 'gender',
      dataType: 'Categorical',
      test: 'Chi-Square',
      pValue: 0.001,
      effectSize: 0.31,
      effectSizeLabel: 'Large',
      correlation: 0.28,
      sensitivityLevel: 'Highly Sensitive',
      groups: ['Male', 'Female'],
      description: 'Gender shows strong correlation with target variable'
    },
    {
      name: 'age_group',
      dataType: 'Categorical', 
      test: 'Chi-Square',
      pValue: 0.018,
      effectSize: 0.18,
      effectSizeLabel: 'Medium',
      correlation: 0.15,
      sensitivityLevel: 'Moderately Sensitive',
      groups: ['Young', 'Middle', 'Senior'],
      description: 'Age group shows moderate correlation with outcomes'
    },
    {
      name: 'race',
      dataType: 'Categorical',
      test: 'Chi-Square', 
      pValue: 0.003,
      effectSize: 0.25,
      effectSizeLabel: 'Large',
      correlation: 0.22,
      sensitivityLevel: 'Highly Sensitive',
      groups: ['White', 'Black', 'Hispanic', 'Asian'],
      description: 'Race shows significant correlation with predictions'
    }
  ],
  summary: {
    totalDetected: 3,
    highlySensitiveCount: 2,
    moderatelySensitiveCount: 1,
    riskLevel: 'HIGH',
  }
};

// Mock Fairness Metrics by Sensitive Attribute
export const mockFairnessMetricsByAttribute = {
  gender: {
    attribute: 'gender',
    groups: ['Male', 'Female'],
    metrics: [
      {
        name: 'Statistical Parity',
        value: 0.23,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'Male': 0.65, 'Female': 0.42 },
        description: '23% difference in positive predictions between genders',
        tooltip: 'Systematically favors males over females',
      },
      {
        name: 'Disparate Impact Ratio',
        value: 0.64,
        status: 'Biased',
        threshold: 0.80,
        groupRates: { 'Male': 0.65, 'Female': 0.42 },
        description: 'Ratio of positive rates between genders',
        tooltip: 'Should be close to 1.0 for fairness',
      },
      {
        name: 'Equal Opportunity',
        value: 0.18,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'Male': 0.72, 'Female': 0.54 },
        description: 'Difference in true positive rates',
        tooltip: 'Equal opportunity for qualified candidates',
      },
      {
        name: 'Equalized Odds',
        value: 0.21,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'Male': 0.68, 'Female': 0.47 },
        description: 'Average difference in TPR and FPR',
        tooltip: 'Balances both true and false positive rates',
      },
      {
        name: 'Calibration',
        value: 0.15,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'Male': 0.78, 'Female': 0.63 },
        description: 'Difference in prediction calibration',
        tooltip: 'Predictions should be equally reliable',
      },
      {
        name: 'Generalized Entropy Index',
        value: 0.089,
        status: 'Fair',
        threshold: 0.10,
        groupRates: { 'Male': 0.52, 'Female': 0.48 },
        description: 'Overall distribution fairness',
        tooltip: 'Measures inequality in outcomes',
      },
    ],
    overallScore: 42,
    overallRating: 'Poor Fairness',
    riskClassification: 'High Bias Risk',
    primaryIssues: ['Gender discrimination'],
  },
  
  age_group: {
    attribute: 'age_group',
    groups: ['Young', 'Middle', 'Senior'],
    metrics: [
      {
        name: 'Statistical Parity',
        value: 0.15,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'Young': 0.68, 'Middle': 0.52, 'Senior': 0.38 },
        description: '15% max difference in positive predictions between age groups',
        tooltip: 'Favors younger applicants',
      },
      {
        name: 'Disparate Impact Ratio',
        value: 0.56,
        status: 'Biased',
        threshold: 0.80,
        groupRates: { 'Young': 0.68, 'Middle': 0.52, 'Senior': 0.38 },
        description: 'Ratio of positive rates between age groups',
        tooltip: 'Senior group significantly disadvantaged',
      },
      {
        name: 'Equal Opportunity',
        value: 0.12,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'Young': 0.75, 'Middle': 0.63, 'Senior': 0.51 },
        description: 'Difference in true positive rates',
        tooltip: 'Age-based opportunity disparity',
      },
      {
        name: 'Equalized Odds',
        value: 0.14,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'Young': 0.72, 'Middle': 0.58, 'Senior': 0.49 },
        description: 'Average difference in TPR and FPR',
        tooltip: 'Consistent age bias across metrics',
      },
      {
        name: 'Calibration',
        value: 0.09,
        status: 'Fair',
        threshold: 0.10,
        groupRates: { 'Young': 0.81, 'Middle': 0.72, 'Senior': 0.74 },
        description: 'Difference in prediction calibration',
        tooltip: 'Relatively well calibrated across ages',
      },
      {
        name: 'Generalized Entropy Index',
        value: 0.108,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'Young': 0.45, 'Middle': 0.35, 'Senior': 0.20 },
        description: 'Overall distribution fairness',
        tooltip: 'Unequal outcome distribution by age',
      },
    ],
    overallScore: 58,
    overallRating: 'Moderate Fairness',
    riskClassification: 'Medium Bias Risk',
    primaryIssues: ['Age discrimination', 'Senior disadvantage'],
  },

  race: {
    attribute: 'race',
    groups: ['White', 'Black', 'Hispanic', 'Asian'],
    metrics: [
      {
        name: 'Statistical Parity',
        value: 0.29,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'White': 0.64, 'Black': 0.35, 'Hispanic': 0.41, 'Asian': 0.59 },
        description: '29% max difference in positive predictions between races',
        tooltip: 'Significant racial disparities in outcomes',
      },
      {
        name: 'Disparate Impact Ratio',
        value: 0.55,
        status: 'Biased',
        threshold: 0.80,
        groupRates: { 'White': 0.64, 'Black': 0.35, 'Hispanic': 0.41, 'Asian': 0.59 },
        description: 'Ratio of positive rates between racial groups',
        tooltip: 'Black applicants severely disadvantaged',
      },
      {
        name: 'Equal Opportunity',
        value: 0.25,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'White': 0.71, 'Black': 0.46, 'Hispanic': 0.52, 'Asian': 0.68 },
        description: 'Difference in true positive rates',
        tooltip: 'Unequal opportunity across racial groups',
      },
      {
        name: 'Equalized Odds',
        value: 0.23,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'White': 0.69, 'Black': 0.42, 'Hispanic': 0.48, 'Asian': 0.65 },
        description: 'Average difference in TPR and FPR',
        tooltip: 'Consistent racial bias pattern',
      },
      {
        name: 'Calibration',
        value: 0.18,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'White': 0.79, 'Black': 0.61, 'Hispanic': 0.66, 'Asian': 0.76 },
        description: 'Difference in prediction calibration',
        tooltip: 'Model less reliable for minority groups',
      },
      {
        name: 'Generalized Entropy Index',
        value: 0.142,
        status: 'Biased',
        threshold: 0.10,
        groupRates: { 'White': 0.40, 'Black': 0.20, 'Hispanic': 0.25, 'Asian': 0.35 },
        description: 'Overall distribution fairness',
        tooltip: 'High inequality in racial outcomes',
      },
    ],
    overallScore: 31,
    overallRating: 'Poor Fairness',
    riskClassification: 'High Bias Risk',
    primaryIssues: ['Racial discrimination', 'Minority disadvantage', 'Systemic bias'],
  }
};

// Mock Mitigation Strategies with Multi-Attribute Support
export const mockMitigationStrategies = {
  strategies: [
    {
      name: 'Reweighing',
      category: 'Preprocessing',
      fairnessImprovement: 38,
      accuracyImpact: -1.8,
      precisionImpact: -1.2,
      f1Impact: -1.5,
      stars: 3,
      recommendation: 'Highly Recommended',
      description: 'Rebalances training data to reduce bias across all sensitive attributes',
      targetAttributes: ['gender', 'age_group', 'race']
    },
    {
      name: 'Resampling (SMOTE)',
      category: 'Preprocessing', 
      fairnessImprovement: 32,
      accuracyImpact: -3.1,
      precisionImpact: -2.4,
      f1Impact: -2.8,
      stars: 2,
      recommendation: 'Good option',
      description: 'Synthetic oversampling to balance protected groups',
      targetAttributes: ['gender', 'age_group', 'race']
    },
    {
      name: 'Adversarial Debiasing',
      category: 'In-processing',
      fairnessImprovement: 35,
      accuracyImpact: -2.9,
      precisionImpact: -2.1,
      f1Impact: -2.5,
      stars: 3,
      recommendation: 'Recommended',
      description: 'Neural network with adversarial training for each sensitive attribute',
      targetAttributes: ['gender', 'age_group', 'race']
    },
    {
      name: 'Fair Representation Learning',
      category: 'In-processing',
      fairnessImprovement: 29,
      accuracyImpact: -2.3,
      precisionImpact: -1.8,
      f1Impact: -2.0,
      stars: 2,
      recommendation: 'Good option',
      description: 'Learns fair representations independent of sensitive attributes',
      targetAttributes: ['gender', 'age_group', 'race']
    },
    {
      name: 'Threshold Optimization',
      category: 'Post-processing',
      fairnessImprovement: 25,
      accuracyImpact: -0.9,
      precisionImpact: -1.1,
      f1Impact: -1.0,
      stars: 3,
      recommendation: 'Minimal Accuracy Loss',
      description: 'Optimizes decision thresholds for each protected group',
      targetAttributes: ['gender', 'age_group', 'race']
    },
    {
      name: 'Calibration Post-processing',
      category: 'Post-processing',
      fairnessImprovement: 22,
      accuracyImpact: -0.5,
      precisionImpact: -0.8,
      f1Impact: -0.6,
      stars: 2,
      recommendation: 'Conservative approach',
      description: 'Adjusts predictions to ensure equal calibration across groups',
      targetAttributes: ['gender', 'age_group', 'race']
    }
  ]
};

// Mock Before/After Comparison Data by Attribute
export const mockBeforeAfterComparison = {
  gender: {
    attribute: 'gender',
    groups: ['Male', 'Female'],
    before: { Male: 65, Female: 42 },
    after: { Male: 58, Female: 55 },
    improvement: { Male: -7, Female: 13 }
  },
  age_group: {
    attribute: 'age_group', 
    groups: ['Young', 'Middle', 'Senior'],
    before: { Young: 68, Middle: 52, Senior: 38 },
    after: { Young: 62, Middle: 55, Senior: 48 },
    improvement: { Young: -6, Middle: 3, Senior: 10 }
  },
  race: {
    attribute: 'race',
    groups: ['White', 'Black', 'Hispanic', 'Asian'],
    before: { White: 64, Black: 35, Hispanic: 41, Asian: 59 },
    after: { White: 58, Black: 52, Hispanic: 53, Asian: 57 },
    improvement: { White: -6, Black: 17, Hispanic: 12, Asian: -2 }
  }
};

// Utility Functions for Backend Compatibility
export const getAttributeDisplayName = (attribute) => {
  const displayNames = {
    gender: 'Gender',
    age_group: 'Age Group',
    age: 'Age', 
    race: 'Race',
    education_years: 'Education Years',
    work_experience: 'Work Experience',
    income: 'Income',
    marital_status: 'Marital Status'
  };
  return displayNames[attribute] || attribute.charAt(0).toUpperCase() + attribute.slice(1).replace(/_/g, ' ');
};

export const getGroupDisplayNames = (attribute) => {
  const groups = {
    gender: { Male: 'Male', Female: 'Female' },
    age_group: { Young: 'Young (18-35)', Middle: 'Middle (36-55)', Senior: 'Senior (55+)' },
    race: { White: 'White', Black: 'Black', Hispanic: 'Hispanic', Asian: 'Asian' }
  };
  return groups[attribute] || {};
};

// Mock API Functions (for future backend integration)
export const mockAPI = {
  getSensitiveFeatures: () => Promise.resolve(mockSensitiveFeatures),
  getFairnessMetrics: (attribute = 'gender') => Promise.resolve(mockFairnessMetricsByAttribute[attribute]),
  getAllFairnessMetrics: () => Promise.resolve(mockFairnessMetricsByAttribute),
  getMitigationStrategies: () => Promise.resolve(mockMitigationStrategies),
  getBeforeAfterComparison: (attribute = 'gender') => Promise.resolve(mockBeforeAfterComparison[attribute]),
  getAllBeforeAfterComparisons: () => Promise.resolve(mockBeforeAfterComparison)
};
