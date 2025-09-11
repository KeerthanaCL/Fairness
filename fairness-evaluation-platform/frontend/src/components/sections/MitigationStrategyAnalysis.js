import React, { useState } from 'react';
import {
  Star,
  TrendingUp,
  TrendingDown,
} from 'lucide-react';
import FairnessRadarChart from '../visualizations/FairnessRadarChart';
import ComparisonBarChart from '../visualizations/ComparisonBarChart';
import { 
  mockMitigationStrategies,
  mockBeforeAfterComparison,
  getAttributeDisplayName,
  SENSITIVE_ATTRIBUTES 
} from '../../services/mockDataService';

const MitigationStrategyAnalysis = ({ data }) => {
  // React hooks must be called first, before any conditional logic
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedStrategy, setSelectedStrategy] = useState('Adversarial Debiasing');
  const [selectedAttribute, setSelectedAttribute] = useState(SENSITIVE_ATTRIBUTES.GENDER);

  // Debug: Log the incoming data
  console.log('MitigationStrategyAnalysis received data:', data);

  // Use real data if available, fallback to mock data
  const { strategies = [] } = data || mockMitigationStrategies;
  const attributeComparison = mockBeforeAfterComparison[SENSITIVE_ATTRIBUTES.GENDER];

  // Debug: Log the strategies array
  console.log('Strategies array:', strategies);

  // Filter out any undefined or invalid strategies (do this before useEffect)
  const validStrategies = Array.isArray(strategies) ? strategies.filter(strategy => strategy && strategy.name) : [];

  // Update selected strategy when valid strategies change - must be called before any returns
  React.useEffect(() => {
    if (validStrategies.length > 0 && !validStrategies.find(s => s.name === selectedStrategy)) {
      setSelectedStrategy(validStrategies[0].name);
    }
  }, [validStrategies, selectedStrategy]);

  // Additional safety check for strategies array
  if (!Array.isArray(strategies)) {
    console.error('MitigationStrategyAnalysis: strategies should be an array', data);
    return (
      <div className="p-6">
        <h2 className="text-xl font-bold text-red-600 dark:text-red-400">
          Error loading mitigation strategies data
        </h2>
      </div>
    );
  }

  if (validStrategies.length === 0) {
    return (
      <div className="p-6">
        <h2 className="text-xl font-bold text-yellow-600 dark:text-yellow-400">
          No mitigation strategies available
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Please ensure the analysis has completed successfully.
        </p>
      </div>
    );
  }

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const handleAttributeChange = (event) => {
    setSelectedAttribute(event.target.value);
  };

  const getRecommendationColor = (recommendation) => {
    if (recommendation.includes('Highly Recommended')) return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
    if (recommendation.includes('Recommended')) return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
    return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
  };

  const getImpactColor = (impact) => {
    if (impact > 0) return 'text-green-600 dark:text-green-400';
    if (impact < -5) return 'text-red-600 dark:text-red-400';
    return 'text-yellow-600 dark:text-yellow-400';
  };

  const renderStars = (count) => {
    return Array.from({ length: 3 }, (_, i) => (
      <Star
        key={i}
        className={`w-4 h-4 ${i < count ? 'text-yellow-500 fill-current' : 'text-gray-300'}`}
      />
    ));
  };

  const getCategoryStrategies = (category) => {
    return validStrategies.filter(s => s.category === category);
  };

  const bestStrategies = {
    'Preprocessing': getCategoryStrategies('Preprocessing').sort((a, b) => b.fairnessImprovement - a.fairnessImprovement)[0],
    'In-processing': getCategoryStrategies('In-processing').sort((a, b) => b.fairnessImprovement - a.fairnessImprovement)[0],
    'Post-processing': getCategoryStrategies('Post-processing').sort((a, b) => b.fairnessImprovement - a.fairnessImprovement)[0],
  };

  // Mock visualization data - now dynamic based on selected attribute
  const mockVisualizationData = {
    radar: {
      before: [42, 38, 45, 35, 48, 41],
      after: [78, 72, 81, 69, 75, 77],
      metrics: ['Statistical Parity', 'Disparate Impact', 'Equal Opportunity', 'Equalized Odds', 'Calibration', 'Entropy Index'],
    },
    performance: {
      before: { accuracy: 85.2, precision: 82.1, recall: 79.8, f1: 80.9 },
      after: { accuracy: 83.4, precision: 80.9, recall: 78.6, f1: 79.4 },
    },
    groupComparison: attributeComparison || {
      before: { Male: 65, Female: 42 },
      after: { Male: 58, Female: 55 },
    },
  };

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6">
        Mitigation Strategy Analysis
      </h1>

      {/* Strategy Performance Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 mb-8">
        <div className="p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Strategy Performance Comparison
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Strategy</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Category</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Fairness Improvement</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Accuracy Impact</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Precision Impact</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">F1 Impact</th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Recommendation</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {validStrategies.map((strategy) => (
                <tr
                  key={strategy.name}
                  className={`hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors ${
                    selectedStrategy === strategy.name ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                  }`}
                  onClick={() => setSelectedStrategy(strategy.name)}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {strategy.name}
                      </div>
                      {renderStars(strategy.stars)}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="inline-flex px-2 py-1 text-xs font-medium border border-blue-300 dark:border-blue-600 rounded text-blue-700 dark:text-blue-300">
                      {strategy.category}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    <span className="text-sm font-bold text-green-600 dark:text-green-400">
                      <div className="flex items-center justify-center gap-1">
                        <TrendingUp className="w-4 h-4" />
                        +{strategy.fairnessImprovement}%
                      </div>
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    <span className={`text-sm font-bold ${getImpactColor(strategy.accuracyImpact)}`}>
                      <div className="flex items-center justify-center gap-1">
                        <TrendingDown className="w-4 h-4" />
                        {strategy.accuracyImpact}%
                      </div>
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    <span className={`text-sm font-bold ${getImpactColor(strategy.precisionImpact)}`}>
                      <div className="flex items-center justify-center gap-1">
                        <TrendingDown className="w-4 h-4" />
                        {strategy.precisionImpact}%
                      </div>
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    <span className={`text-sm font-bold ${getImpactColor(strategy.f1Impact)}`}>
                      <div className="flex items-center justify-center gap-1">
                        <TrendingDown className="w-4 h-4" />
                        {strategy.f1Impact}%
                      </div>
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getRecommendationColor(strategy.recommendation)}`}>
                      {strategy.recommendation}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Best Performers by Category */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {Object.entries(bestStrategies).map(([category, strategy]) => (
          <div key={category} className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-blue-600 dark:text-blue-400 mb-2">
              Best {category}
            </h3>
            <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              {strategy.name}
            </h4>
            <div className="flex items-center gap-2 mb-3">
              {renderStars(strategy.stars)}
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {strategy.recommendation}
              </span>
            </div>
            <div className="text-sm font-bold text-green-600 dark:text-green-400">
              +{strategy.fairnessImprovement}% Fairness Improvement
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {strategy.accuracyImpact}% Accuracy Impact
            </div>
          </div>
        ))}
      </div>

      {/* Before/After Model Comparison Visualizations */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
          Before/After Model Comparison: {selectedStrategy}
        </h3>
        
        <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
          <nav className="-mb-px flex space-x-8">
            {['Overview', 'Fairness Metrics', 'Performance Impact', 'Group Analysis'].map((tab, index) => (
              <button
                key={tab}
                onClick={() => handleTabChange(null, index)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  selectedTab === index
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>

        {selectedTab === 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                Current Model Status
              </h4>
              <div className="text-3xl font-bold text-red-600 dark:text-red-400">
                42/100
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Poor Fairness Score
              </div>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
              <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                After {selectedStrategy}
              </h4>
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                76/100
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Good Fairness Score
              </div>
            </div>
          </div>
        )}

        {selectedTab === 1 && (
          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-6">
              Fairness metrics comparison showing improvement across all dimensions:
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Radar Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Visual Comparison
                </h4>
                <FairnessRadarChart data={mockVisualizationData.radar} />
              </div>
              
              {/* Tabular Comparison */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Detailed Metrics
                </h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Metric
                        </th>
                        <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Before
                        </th>
                        <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          After
                        </th>
                        <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Improvement
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                      {mockVisualizationData.radar.metrics.map((metric, index) => {
                        const before = mockVisualizationData.radar.before[index];
                        const after = mockVisualizationData.radar.after[index];
                        const improvement = after - before;
                        return (
                          <tr key={metric}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                              {metric}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-center">
                              <span className="text-sm font-bold text-red-600 dark:text-red-400">
                                {before}/100
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-center">
                              <span className="text-sm font-bold text-green-600 dark:text-green-400">
                                {after}/100
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-center">
                              <span className={`text-sm font-bold ${improvement > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                                +{improvement}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 2 && (
          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-6">
              Performance metrics showing the trade-off between fairness and accuracy:
            </p>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-700">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Performance Metric
                      </th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Before Mitigation
                      </th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        After Mitigation
                      </th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Change
                      </th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Impact
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                    {Object.entries(mockVisualizationData.performance.before).map(([metric, beforeValue]) => {
                      const afterValue = mockVisualizationData.performance.after[metric];
                      const change = afterValue - beforeValue;
                      const changePercent = ((change / beforeValue) * 100);
                      
                      return (
                        <tr key={metric}>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className="text-sm font-semibold text-gray-900 dark:text-white capitalize">
                              {metric.replace(/([A-Z])/g, ' $1').trim()}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-center">
                            <span className="text-sm font-bold text-gray-900 dark:text-white">
                              {beforeValue.toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-center">
                            <span className="text-sm font-bold text-gray-900 dark:text-white">
                              {afterValue.toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-center">
                            <span className={`text-sm font-bold ${change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                              {change >= 0 ? '+' : ''}{change.toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-center">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              Math.abs(changePercent) < 2 
                                ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' 
                                : Math.abs(changePercent) < 5 
                                ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' 
                                : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                            }`}>
                              {Math.abs(changePercent) < 2 ? 'Minimal' : Math.abs(changePercent) < 5 ? 'Moderate' : 'Significant'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-700 p-4 mt-4">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <span className="font-semibold">Impact Assessment:</span> The {selectedStrategy} strategy shows minimal performance degradation 
                  while significantly improving fairness metrics. This represents an excellent trade-off for bias mitigation.
                </p>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 3 && (
          <div>
            <div className="flex justify-between items-center mb-6">
              <p className="text-gray-700 dark:text-gray-300">
                Group-wise prediction rates showing reduced disparity:
              </p>
              
              {/* Sensitive Attribute Selector for Group Analysis */}
              <div className="min-w-[200px]">
                <select
                  value={selectedAttribute}
                  onChange={handleAttributeChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  {Object.values(SENSITIVE_ATTRIBUTES).map((attr) => (
                    <option key={attr} value={attr}>
                      {getAttributeDisplayName(attr)}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            
            {/* Show comparison for selected attribute */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 mb-4">
              <h4 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                {getAttributeDisplayName(selectedAttribute)} Analysis
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Mitigation impact on {getAttributeDisplayName(selectedAttribute)} groups using {selectedStrategy} strategy
              </p>
            </div>
            
            <ComparisonBarChart data={mockVisualizationData.groupComparison} />
            
            {/* Multi-Attribute Summary */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg shadow-md p-6 mt-6">
              <h4 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                Multi-Attribute Mitigation Summary
              </h4>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                The <span className="font-bold">{selectedStrategy}</span> strategy addresses bias across all detected sensitive attributes:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.values(SENSITIVE_ATTRIBUTES).map((attr) => {
                  const attrData = mockBeforeAfterComparison[attr];
                  if (!attrData) return null;
                  
                  const groups = Object.keys(attrData.before);
                  const maxDiffBefore = Math.max(...Object.values(attrData.before)) - Math.min(...Object.values(attrData.before));
                  const maxDiffAfter = Math.max(...Object.values(attrData.after)) - Math.min(...Object.values(attrData.after));
                  const improvement = ((maxDiffBefore - maxDiffAfter) / maxDiffBefore * 100).toFixed(1);
                  
                  return (
                    <div key={attr} className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 h-full">
                      <h5 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                        {getAttributeDisplayName(attr)}
                      </h5>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {groups.length} groups analyzed
                      </p>
                      <div className="text-lg font-bold text-green-600 dark:text-green-400">
                        {improvement}% improvement
                      </div>
                      <p className="text-xs text-gray-500 dark:text-gray-500">
                        Disparity reduction
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MitigationStrategyAnalysis;
