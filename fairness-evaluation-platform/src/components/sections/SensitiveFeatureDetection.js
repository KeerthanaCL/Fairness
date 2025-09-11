import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Brain,
  BarChart3,
  AlertTriangle,
  AlertCircle,
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
    if (level.includes('Highly')) return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
    if (level.includes('Moderately')) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
    return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
  };

  const getSensitivityIcon = (level) => {
    if (level.includes('Highly')) return <AlertCircle className="text-red-500" />;
    if (level.includes('Moderately')) return <AlertTriangle className="text-yellow-500" />;
    return <BarChart3 className="text-blue-500" />;
  };

  const getRiskLevelColor = (level) => {
    if (level === 'HIGH') return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
    if (level === 'MEDIUM') return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
    return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
  };

  const getEffectSizeColor = (size) => {
    if (size === 'Large') return 'text-red-500';
    if (size === 'Medium') return 'text-yellow-500';
    return 'text-green-500';
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
              {summary.totalDetected}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Sensitive Features Detected
            </p>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-6 text-center">
            <div className="text-3xl font-bold text-red-600 dark:text-red-400 mb-2">
              {summary.highlySensitiveCount}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Highly Sensitive (p &lt; 0.01)
            </p>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-6 text-center">
            <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-400 mb-2">
              {summary.moderatelySensitiveCount}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Moderately Sensitive (0.01 ≤ p &lt; 0.05)
            </p>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-6 text-center">
            <span className={`inline-flex px-4 py-2 text-base font-medium rounded-full ${getRiskLevelColor(summary.riskLevel)}`}>
              {summary.riskLevel} RISK
            </span>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Model Risk Level
            </p>
          </div>
        </div>
      </div>

      {/* Detailed Feature Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Detected Sensitive Features
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
            Features flagged by statistical testing (showing only significant results)
          </p>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Feature Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Data Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Test</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">P-Value</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Effect Size</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Correlation</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Sensitivity Level</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Groups</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {detectedFeatures.map((feature) => (
                <React.Fragment key={feature.name}>
                  <tr className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 dark:text-gray-100 capitalize">
                        {feature.name.replace('_', ' ')}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="inline-flex px-2 py-1 text-xs font-medium border border-gray-300 dark:border-gray-600 rounded text-gray-700 dark:text-gray-300">
                        {feature.dataType}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">{feature.test}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <span className={`text-sm font-bold ${
                        feature.pValue < 0.01 ? 'text-red-600 dark:text-red-400' : 'text-yellow-600 dark:text-yellow-400'
                      }`}>
                        {feature.pValue.toFixed(3)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <div>
                        <div className={`text-sm font-bold ${getEffectSizeColor(feature.effectSizeLabel)}`}>
                          {feature.effectSize.toFixed(2)}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          ({feature.effectSizeLabel})
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                        {feature.correlation.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <span className={`inline-flex items-center px-2 py-1 text-xs font-medium rounded-full ${getSensitivityColor(feature.sensitivityLevel)}`}>
                        {getSensitivityIcon(feature.sensitivityLevel)}
                        <span className="ml-1">{feature.sensitivityLevel}</span>
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <div className="text-sm text-gray-900 dark:text-gray-100">
                        {feature.groups.length} groups
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        ({feature.groups.join(', ')})
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <button
                        onClick={() => handleRowExpansion(feature.name)}
                        className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                        title="Expand for detailed analysis"
                      >
                        {expandedRows[feature.name] ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </button>
                      <button className="p-1 ml-1 text-blue-500 hover:text-blue-700 transition-colors" title="AI Explanation">
                        <Brain className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                  
                  {/* Expanded Row Details */}
                  <tr>
                    <td colSpan={9} className="px-6 py-0">
                      {expandedRows[feature.name] && (
                        <div className="py-4">
                          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4">
                            <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
                              Detailed Analysis for {feature.name.replace('_', ' ').toUpperCase()}
                            </h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div>
                                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
                                  Statistical Test Results:
                                </p>
                                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                                  • {feature.test} test p-value: {feature.pValue.toFixed(3)}
                                </p>
                                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                                  • Effect size ({feature.test === 'Chi-Square' ? "Cramér's V" : "Eta-squared"}): {feature.effectSize.toFixed(3)} ({feature.effectSizeLabel})
                                </p>
                                <p className="text-sm text-gray-600 dark:text-gray-400">
                                  • Correlation with target: {feature.correlation.toFixed(3)}
                                </p>
                              </div>
                              <div>
                                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
                                  Interpretation:
                                </p>
                                <p className="text-sm text-gray-600 dark:text-gray-400">
                                  {feature.description}
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </td>
                  </tr>
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detection Summary */}
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 mt-8">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Detection Summary & Recommendations
        </h2>
        <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
          <span className="font-medium">Risk Assessment:</span> Your model shows <strong>{summary.riskLevel}</strong> bias risk 
          with {summary.highlySensitiveCount} highly sensitive and {summary.moderatelySensitiveCount} moderately 
          sensitive features detected.
        </p>
        <p className="text-sm text-gray-700 dark:text-gray-300">
          <span className="font-medium">Next Steps:</span> Proceed to fairness metrics analysis to understand specific bias patterns, 
          then explore mitigation strategies to address these issues.
        </p>
      </div>
    </div>
  );
};

export default SensitiveFeatureDetection;
