import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Brain,
  BarChart3,
  AlertTriangle,
  AlertCircle,
  Info,
} from 'lucide-react';
import { mockSensitiveFeatures } from '../../services/mockDataService';

const SensitiveFeatureDetection = ({ data }) => {
  const [expandedRows, setExpandedRows] = useState({});
  
  // Use real data if available, fallback to mock data
  const { detectedFeatures = [], summary = {} } = data || mockSensitiveFeatures;

  const handleRowExpansion = (featureName) => {
    setExpandedRows(prev => ({
      ...prev,
      [featureName]: !prev[featureName]
    }));
  };

  const getSensitivityColor = (level) => {
    const normalizedLevel = level?.toLowerCase() || '';
    if (normalizedLevel.includes('highly')) return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
    if (normalizedLevel.includes('moderately')) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
    return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
  };

  const getSensitivityIcon = (level) => {
    const normalizedLevel = level?.toLowerCase() || '';
    if (normalizedLevel.includes('highly')) return <AlertCircle className="text-red-500" />;
    if (normalizedLevel.includes('moderately')) return <AlertTriangle className="text-yellow-500" />;
    return <BarChart3 className="w-4 h-4 text-blue-500" />;
  };

  const getRiskLevelColor = (level) => {
    const normalizedLevel = level?.toUpperCase() || 'LOW';
    if (normalizedLevel === 'HIGH') return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
    if (normalizedLevel === 'MEDIUM') return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
    return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
  };

  const getEffectSizeColor = (size) => {
    const normalizedSize = size?.toLowerCase() || '';
    if (normalizedSize === 'large') return 'text-red-500';
    if (normalizedSize === 'medium') return 'text-yellow-500';
    return 'text-green-500';
  };

  // Format test name for display
  const formatTestName = (test) => {
    const testMapping = {
      'hsic_nocco': 'HSIC/NOCCO',
      'chi_square': 'Chi-Square',
      'anova': 'ANOVA',
      'pearson': 'Pearson Correlation',
      't_test': 'T-Test'
    };
    return testMapping[test?.toLowerCase()] || test;
  };

  // Check if using HSIC method
  const isHsicMethod = (test) => {
    return test?.toLowerCase().includes('hsic');
  };

  // Format sensitivity level for display
  const formatSensitivityLevel = (level) => {
    return level?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || 'Low Sensitivity';
  };

  // Get metric label based on test type
  const getMetricLabel = (test) => {
    if (isHsicMethod(test)) {
      return 'NOCCO Score';
    }
    if (test?.toLowerCase() === 'chi_square') {
      return "Cramér's V";
    }
    if (test?.toLowerCase() === 'anova') {
      return 'Eta-squared';
    }
    return 'Effect Size';
  };

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6">
        Sensitive Feature Detection Results
      </h1>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-6 text-center">
            <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
              {summary.totalDetected || 0}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Sensitive Features Detected
            </p>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-6 text-center">
            <div className="text-3xl font-bold text-red-600 dark:text-red-400 mb-2">
              {summary.highlySensitiveCount || 0}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Highly Sensitive
            </p>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-6 text-center">
            <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-400 mb-2">
              {summary.moderatelySensitiveCount || 0}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Moderately Sensitive
            </p>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-6 text-center">
            <span className={`inline-flex px-4 py-2 text-base font-medium rounded-full ${getRiskLevelColor(summary.riskLevel)}`}>
              {summary.riskLevel?.toUpperCase() || 'LOW'} RISK
            </span>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Model Risk Level
            </p>
          </div>
        </div>
      </div>

      {/* Method Info Banner */}
      {detectedFeatures.length > 0 && isHsicMethod(detectedFeatures[0]?.test) && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6 flex items-start">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 mr-3 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-1">
              HSIC/NOCCO Detection Method
            </p>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Features are ranked by NOCCO (Normalized Cross-Covariance Operator) scores. 
              Higher scores indicate stronger statistical dependence with the target variable, 
              which may lead to disparate outcomes. Threshold is automatically determined using 
              the median NOCCO value across all features.
            </p>
          </div>
        </div>
      )}

      {/* Detailed Feature Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Detected Sensitive Features
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
            Features flagged by statistical analysis (sorted by siginficance)
          </p>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Feature Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Data Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Test Method</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">P-Value</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Effect Size</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Correlation</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Sensitivity Level</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Groups</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {detectedFeatures.length === 0 ? (
                <tr>
                  <td colSpan={9} className="px-6 py-8 text-center">
                    <p className="text-gray-500 dark:text-gray-400">
                      No sensitive features detected in the dataset.
                    </p>
                  </td>
                </tr>
              ) : (
                detectedFeatures.map((feature) => (
                  <React.Fragment key={feature.name}>
                    <tr className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900 dark:text-gray-100 capitalize">
                          {feature.name.replace(/_/g, ' ')}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="inline-flex px-2 py-1 text-xs font-medium border border-gray-300 dark:border-gray-600 rounded text-gray-700 dark:text-gray-300 capitalize">
                          {feature.dataType}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <span className="text-sm text-gray-900 dark:text-gray-100">
                            {formatTestName(feature.test)}
                          </span>
                          {isHsicMethod(feature.test) && (
                            <div className="text-xs text-blue-600 dark:text-blue-400 font-semibold mt-1">
                              (HSIC/NOCCO)
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span className={`text-sm font-bold ${
                          feature.pValue < 0.01 ? 'text-red-600 dark:text-red-400' : 
                          feature.pValue < 0.05 ? 'text-yellow-600 dark:text-yellow-400' :
                          'text-gray-600 dark:text-gray-400'
                        }`}>
                          {feature.pValue.toFixed(4)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div>
                          <div className={`text-sm font-bold ${getEffectSizeColor(feature.effectSizeLabel)}`}>
                            {feature.effectSize.toFixed(3)}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 capitalize">
                            ({feature.effectSizeLabel})
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div>
                          <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                            {feature.correlation.toFixed(3)}
                          </span>
                          {isHsicMethod(feature.test) && (
                            <div className="text-xs text-gray-500 dark:text-gray-400">
                              NOCCO
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span className={`inline-flex items-center px-2 py-1 text-xs font-medium rounded-full ${getSensitivityColor(feature.sensitivityLevel)}`}>
                          {getSensitivityIcon(feature.sensitivityLevel)}
                          <span className="ml-1 capitalize">
                            {formatSensitivityLevel(feature.sensitivityLevel).replace('Sensitivity', '').trim()}
                          </span>
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div className="text-sm text-gray-900 dark:text-gray-100">
                          {feature.groups?.length || 0} groups
                        </div>
                        {feature.groups && feature.groups.length > 0 && (
                          <div className="text-xs text-gray-500 dark:text-gray-400 truncate max-w-xs">
                            {feature.groups.slice(0, 2).join(', ')}
                            {feature.groups.length > 2 && '...'}
                          </div>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <button
                          onClick={() => handleRowExpansion(feature.name)}
                          className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                          title="Expand for detailed analysis"
                        >
                          {expandedRows[feature.name] ? (
                            <ChevronUp className="w-4 h-4" />
                          ) : (
                            <ChevronDown className="w-4 h-4" />
                          )}
                        </button>
                        <button 
                          className="p-1 ml-1 text-blue-500 hover:text-blue-700 transition-colors" 
                          title="AI Explanation"
                        >
                          <Brain className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  
                  {/* Expanded Row Details */}
                  {expandedRows[feature.name] && (
                      <tr>
                        <td colSpan={9} className="px-6 py-0">
                          <div className="py-4">
                            <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4">
                              <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
                                Detailed Analysis for {feature.name.replace(/_/g, ' ').toUpperCase()}
                              </h4>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
                                    Statistical Test Results:
                                  </p>
                                  <ul className="space-y-1">
                                    <li className="text-sm text-gray-600 dark:text-gray-400">
                                      • Test: {formatTestName(feature.test)}
                                    </li>
                                    <li className="text-sm text-gray-600 dark:text-gray-400">
                                      • P-value: {feature.pValue.toFixed(6)}
                                    </li>
                                    <li className="text-sm text-gray-600 dark:text-gray-400">
                                      • {getMetricLabel(feature.test)}: {feature.effectSize.toFixed(4)} ({feature.effectSizeLabel})
                                    </li>
                                    <li className="text-sm text-gray-600 dark:text-gray-400">
                                      • Correlation: {feature.correlation.toFixed(4)}
                                    </li>
                                    <li className="text-sm text-gray-600 dark:text-gray-400">
                                      • Sensitivity: {formatSensitivityLevel(feature.sensitivityLevel)}
                                    </li>
                                  </ul>
                                   {isHsicMethod(feature.test) && (
                                      <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
                                        <p className="text-xs font-semibold text-blue-900 dark:text-blue-300 mb-1">
                                          HSIC/NOCCO Method Explanation:
                                        </p>
                                        <p className="text-xs text-blue-800 dark:text-blue-400">
                                          NOCCO (Normalized Cross-Covariance Operator) measures the statistical dependence between 
                                          this feature and the target variable. Features with NOCCO scores above the median threshold 
                                          are marked as sensitive. Higher NOCCO values indicate stronger potential for disparate outcomes.
                                        </p>
                                      </div>
                                    )}
                                  
                                  {feature.groups && feature.groups.length > 0 && (
                                    <div className="mt-3">
                                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-1">
                                        Groups:
                                      </p>
                                      <div className="flex flex-wrap gap-1">
                                        {feature.groups.map((group, idx) => (
                                          <span 
                                            key={idx}
                                            className={`inline-flex px-2 py-1 text-xs rounded ${
                                              idx === 0 
                                                ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 font-medium'
                                                : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                                            }`}
                                          >
                                            {group}
                                            {idx === 0 && ' (most affected)'}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
                                    Interpretation:
                                  </p>
                                  <p className="text-sm text-gray-600 dark:text-gray-400">
                                    {feature.description}
                                  </p>
                                  
                                  {isHsicMethod(feature.test) && (
                                    <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
                                      <p className="text-xs text-gray-700 dark:text-gray-300 mb-2">
                                        <strong>HSIC/NOCCO Analysis:</strong>
                                      </p>
                                      <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                                        <li>• NOCCO Score: {feature.correlation.toFixed(3)} ({feature.correlation >= 0.5 ? 'Strong' : feature.correlation >= 0.3 ? 'Moderate' : 'Weak'} dependence)</li>
                                        <li>• Interpretation: This feature shows {feature.correlation >= 0.5 ? 'strong' : feature.correlation >= 0.3 ? 'moderate' : 'weak'} statistical dependence with the target</li>
                                        <li>• Impact: May lead to disparate outcomes across different groups</li>
                                        <li>• Action: Review and consider mitigation strategies if bias is a concern</li>
                                      </ul>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detection Summary */}
      {detectedFeatures.length > 0 && (
        <div className={`border rounded-lg p-6 mt-8 ${
          summary.riskLevel?.toUpperCase() === 'HIGH' 
            ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
            : summary.riskLevel?.toUpperCase() === 'MEDIUM'
            ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
            : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
        }`}>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Detection Summary & Recommendations
          </h2>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
            <span className="font-medium">Risk Assessment:</span> Your model shows{' '}
            <strong>{summary.riskLevel?.toUpperCase() || 'LOW'}</strong> bias risk with{' '}
            {summary.highlySensitiveCount || 0} highly sensitive and{' '}
            {summary.moderatelySensitiveCount || 0} moderately sensitive features detected.
          </p>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            <span className="font-medium">Next Steps:</span> {
              summary.riskLevel?.toUpperCase() === 'HIGH' 
                ? 'Immediate attention required. Review the highly sensitive features and consider bias mitigation strategies before deployment.'
                : summary.riskLevel?.toUpperCase() === 'MEDIUM'
                ? 'Proceed with caution. Analyze fairness metrics to understand specific bias patterns and implement mitigation strategies.'
                : 'Low risk detected. Monitor these features during model deployment and continue to assess fairness metrics.'
            }
          </p>
        </div>
      )}
    </div>
  );
};

export default SensitiveFeatureDetection;
