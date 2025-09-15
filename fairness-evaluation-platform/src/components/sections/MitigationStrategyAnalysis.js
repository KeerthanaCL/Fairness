import React, { useState, useEffect } from 'react';
import {
  Star,
  TrendingUp,
  TrendingDown,
  Play,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react';
import FairnessRadarChart from '../visualizations/FairnessRadarChart';
import ComparisonBarChart from '../visualizations/ComparisonBarChart';
import { 
  getAttributeDisplayName
} from '../../services/mockDataService';
import { FairnessAPIService } from '../../services';

const MitigationStrategyAnalysis = ({ data, analysisId }) => {
  // React hooks must be called first, before any conditional logic
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedStrategy, setSelectedStrategy] = useState('Adversarial Debiasing');
  const [selectedAttribute, setSelectedAttribute] = useState(''); // No default hardcoded value
  const [availableAttributes, setAvailableAttributes] = useState([]); // Dynamic sensitive attributes
  const [beforeAfterData, setBeforeAfterData] = useState(null);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [fullApiData, setFullApiData] = useState(null); // Store full API response
  const [attributesError, setAttributesError] = useState(null); // Error state for attributes
  const [comparisonError, setComparisonError] = useState(null); // Error state for comparisons
  
  // Real mitigation states
  const [realMitigationData, setRealMitigationData] = useState(null);
  const [loadingRealMitigation, setLoadingRealMitigation] = useState(false);
  const [realMitigationError, setRealMitigationError] = useState(null);
  const [showRealResults, setShowRealResults] = useState(false);

  // Debug: Log the incoming data
  console.log('MitigationStrategyAnalysis received data:', data);
  console.log('MitigationStrategyAnalysis received analysisId:', analysisId);

  // Use real data, no fallback to mock data
  const { strategies = [] } = data || {};

  // Fetch available sensitive attributes dynamically
  useEffect(() => {
    const fetchSensitiveAttributes = async () => {
      if (!analysisId) {
        setAttributesError('No analysis ID provided. Please start an analysis first.');
        return;
      }

      try {
        setAttributesError(null);
        const sensitiveFeatures = await FairnessAPIService.getSensitiveFeatures(analysisId);
        console.log('Fetched sensitive features for mitigation:', sensitiveFeatures);
        
        if (!sensitiveFeatures?.detectedFeatures || sensitiveFeatures.detectedFeatures.length === 0) {
          setAttributesError('No sensitive features detected in the analysis. Please check your dataset.');
          return;
        }

        const attributes = sensitiveFeatures.detectedFeatures.map(feature => feature.name);
        setAvailableAttributes(attributes);
        
        // Set first available attribute as default if no selection or current selection is invalid
        if (attributes.length > 0 && (!selectedAttribute || !attributes.includes(selectedAttribute))) {
          setSelectedAttribute(attributes[0]);
        }
      } catch (error) {
        console.error('Failed to fetch sensitive attributes for mitigation:', error);
        setAttributesError(`Failed to load sensitive attributes: ${error.message || 'Unknown error'}`);
        setAvailableAttributes([]);
      }
    };

    fetchSensitiveAttributes();
  }, [analysisId]);

  // Fetch before/after comparison data based on selected attribute and strategy
  useEffect(() => {
    const fetchBeforeAfterData = async () => {
      if (!analysisId) {
        setComparisonError('No analysis ID provided. Please start an analysis first.');
        return;
      }

      if (!selectedStrategy) {
        setComparisonError('No mitigation strategy selected.');
        return;
      }

      if (!selectedAttribute) {
        // Wait for attributes to load
        return;
      }

      try {
        setLoadingComparison(true);
        setComparisonError(null);
        const result = await FairnessAPIService.getBeforeAfterComparison(analysisId, selectedStrategy);
        
        // Store the full API response
        setFullApiData(result);
        
        // Debug the API response structure
        console.log('=== Mitigation API Response Debug ===');
        console.log('Full API result:', result);
        console.log('Fairness metrics before:', result?.fairnessMetrics?.before);
        console.log('Fairness metrics after:', result?.fairnessMetrics?.after);
        console.log('Overall score before:', result?.fairnessMetrics?.overallScoreBefore);
        console.log('Overall score after:', result?.fairnessMetrics?.overallScoreAfter);
        console.log('Performance before:', result?.performance?.before);
        console.log('Performance after:', result?.performance?.after);
        console.log('Group comparisons:', result?.groupComparisons);
        console.log('=====================================');
        
        // Extract the group comparison data for the selected attribute
        const groupComparisonData = result?.groupComparisons?.[selectedAttribute];
        console.log('Before/After API result:', result);
        console.log('Group comparison data for', selectedAttribute, ':', groupComparisonData);
        
        if (!result) {
          setComparisonError('No comparison data received from the backend.');
          return;
        }

        if (groupComparisonData) {
          setBeforeAfterData(groupComparisonData);
        } else {
          // Check if the attribute exists in the response
          const availableGroupAttributes = Object.keys(result?.groupComparisons || {});
          if (availableGroupAttributes.length === 0) {
            setComparisonError('No group comparison data available for any attributes.');
          } else {
            setComparisonError(
              `No data available for "${selectedAttribute}". Available attributes: ${availableGroupAttributes.join(', ')}`
            );
          }
          setBeforeAfterData(null);
        }
      } catch (error) {
        console.error('Failed to fetch before/after comparison:', error);
        setComparisonError(`Failed to load comparison data: ${error.message || 'Unknown error'}`);
        setFullApiData(null);
        setBeforeAfterData(null);
      } finally {
        setLoadingComparison(false);
      }
    };

    fetchBeforeAfterData();
  }, [analysisId, selectedStrategy, selectedAttribute]);

  // Real mitigation handler
  const handleApplyRealMitigation = async () => {
    if (!analysisId || !selectedStrategy) {
      setRealMitigationError('Analysis ID and strategy selection required.');
      return;
    }

    setLoadingRealMitigation(true);
    setRealMitigationError(null);
    
    try {
      console.log(`Applying real mitigation: ${selectedStrategy} for analysis ${analysisId}`);
      
      const result = await FairnessAPIService.applyRealMitigation(analysisId, selectedStrategy);
      console.log('Real mitigation result:', result);
      
      setRealMitigationData(result);
      setShowRealResults(true);
      
      // Update the comparison data with real results
      if (result.groupComparisons) {
        const selectedAttrData = result.groupComparisons[selectedAttribute];
        if (selectedAttrData) {
          setBeforeAfterData(selectedAttrData);
        }
      }
      
    } catch (error) {
      console.error('Real mitigation failed:', error);
      setRealMitigationError(`Failed to apply real mitigation: ${error.message || 'Unknown error'}`);
    } finally {
      setLoadingRealMitigation(false);
    }
  };

  // Debug: Log the strategies array
  console.log('Strategies array:', strategies);

  // Filter out any undefined or invalid strategies (do this before useEffect)
  const validStrategies = Array.isArray(strategies) ? strategies.filter(strategy => 
    strategy && 
    typeof strategy === 'object' && 
    strategy.name && 
    typeof strategy.name === 'string' &&
    strategy.category &&
    typeof strategy.fairnessImprovement === 'number'
  ) : [];

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
        <h2 className="text-xl font-bold text-red-600 dark:text-red-400 mb-4">
          Error loading mitigation strategies data
        </h2>
        <p className="text-gray-700 dark:text-gray-300">
          The mitigation strategies data is not in the expected format. Please check the analysis results.
        </p>
      </div>
    );
  }

  if (validStrategies.length === 0) {
    return (
      <div className="p-6">
        <h2 className="text-xl font-bold text-yellow-600 dark:text-yellow-400 mb-4">
          No valid mitigation strategies available
        </h2>
        <p className="text-gray-700 dark:text-gray-300">
          No mitigation strategies were found in the analysis results. Please run the analysis again.
        </p>
      </div>
    );
  }

  // Show error if no analysis ID
  if (!analysisId) {
    return (
      <div className="p-6">
        <h2 className="text-xl font-bold text-red-600 dark:text-red-400 mb-4">
          No Analysis Available
        </h2>
        <p className="text-gray-700 dark:text-gray-300">
          Please start a new analysis by uploading a dataset and running the fairness evaluation.
        </p>
      </div>
    );
  }

  // Show error if attributes failed to load
  if (attributesError) {
    return (
      <div className="p-6">
        <h2 className="text-xl font-bold text-red-600 dark:text-red-400 mb-4">
          Error Loading Sensitive Attributes
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          {attributesError}
        </p>
        <button 
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Reload Page
        </button>
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
    'Preprocessing': getCategoryStrategies('Preprocessing').sort((a, b) => b.fairnessImprovement - a.fairnessImprovement)[0] || null,
    'In-processing': getCategoryStrategies('In-processing').sort((a, b) => b.fairnessImprovement - a.fairnessImprovement)[0] || null,
    'Post-processing': getCategoryStrategies('Post-processing').sort((a, b) => b.fairnessImprovement - a.fairnessImprovement)[0] || null,
  };

  // Real visualization data from API (prioritize real mitigation data when available)
  const visualizationData = (showRealResults && realMitigationData) ? {
    radar: {
      before: realMitigationData.fairnessMetrics?.before || [],
      after: realMitigationData.fairnessMetrics?.after || [],
      metrics: realMitigationData.fairnessMetrics?.metrics || [],
    },
    performance: {
      before: realMitigationData.performance?.before || null,
      after: realMitigationData.performance?.after || null,
    },
    groupComparison: beforeAfterData || null,
    // Add attribute comparison data for multi-attribute summary
    attributeComparison: realMitigationData.groupComparisons || {},
  } : fullApiData ? {
    radar: {
      before: fullApiData.fairnessMetrics?.before || [],
      after: fullApiData.fairnessMetrics?.after || [],
      metrics: fullApiData.fairnessMetrics?.metrics || [],
    },
    performance: {
      before: fullApiData.performance?.before || null,
      after: fullApiData.performance?.after || null,
    },
    groupComparison: beforeAfterData || null,
    // Add attribute comparison data for multi-attribute summary
    attributeComparison: fullApiData.groupComparisons || {},
  } : null;

  // Use backend-calculated overall fairness scores (prioritize real mitigation results)
  const overallFairnessScoreBefore = (showRealResults && realMitigationData) 
    ? realMitigationData?.fairnessMetrics?.overallScoreBefore
    : fullApiData?.fairnessMetrics?.overallScoreBefore;
  const overallFairnessScoreAfter = (showRealResults && realMitigationData)
    ? realMitigationData?.fairnessMetrics?.overallScoreAfter
    : fullApiData?.fairnessMetrics?.overallScoreAfter;

  // Determine fairness score labels
  const getFairnessScoreLabel = (score) => {
    if (score >= 70) return 'Good Fairness Score';
    if (score >= 50) return 'Fair Fairness Score';
    return 'Poor Fairness Score';
  };

  const getFairnessScoreColor = (score) => {
    if (score >= 70) return 'text-green-600 dark:text-green-400';
    if (score >= 50) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Mitigation Strategy Analysis
        </h1>
        
        {/* Real Mitigation Button - Top Level */}
        <div className="flex items-center gap-4">
          {showRealResults && (
            <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
              <CheckCircle className="w-5 h-5" />
              <span className="text-sm font-medium">Real Results Applied</span>
            </div>
          )}
          
          <button
            onClick={handleApplyRealMitigation}
            disabled={loadingRealMitigation || !analysisId || !selectedStrategy}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
              loadingRealMitigation
                ? 'bg-gray-400 text-white cursor-not-allowed'
                : showRealResults
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {loadingRealMitigation ? (
              <>
                <Clock className="w-5 h-5 animate-spin" />
                Processing Real Mitigation...
              </>
            ) : showRealResults ? (
              <>
                <CheckCircle className="w-5 h-5" />
                Re-apply Real Mitigation
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Apply Real Mitigation
              </>
            )}
          </button>
        </div>
      </div>
      
      {/* Real Mitigation Status Messages */}
      {realMitigationError && (
        <div className="mb-6 bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span className="font-medium">Real Mitigation Error:</span>
          </div>
          <p className="text-red-700 dark:text-red-300 mt-1 text-sm">{realMitigationError}</p>
        </div>
      )}
      
      {showRealResults && realMitigationData && (
        <div className="mb-6 bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2 text-green-600 dark:text-green-400 mb-2">
            <CheckCircle className="w-5 h-5" />
            <span className="font-medium">Real Mitigation Applied Successfully!</span>
          </div>
          <p className="text-green-700 dark:text-green-300 text-sm">
            The strategy <strong>"{selectedStrategy}"</strong> has been applied to your model and data. 
            All results below show <strong>actual improvements</strong>, not simulations.
          </p>
        </div>
      )}
      
      {!showRealResults && (
        <div className="mb-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 mb-2">
            <AlertCircle className="w-5 h-5" />
            <span className="font-medium">Strategy Recommendations Available</span>
          </div>
          <p className="text-blue-700 dark:text-blue-300 text-sm">
            Review the mitigation strategies below, select one, then click <strong>"Apply Real Mitigation"</strong> above to see actual improvements instead of simulated results.
          </p>
        </div>
      )}

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
        {Object.entries(bestStrategies)
          .filter(([category, strategy]) => strategy !== null) // Filter out null strategies
          .map(([category, strategy]) => (
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
          {showRealResults ? `Real Results: ${selectedStrategy}` : `Strategy Analysis: ${selectedStrategy}`}
          {showRealResults && (
            <span className="ml-2 inline-flex items-center gap-1 text-green-600 dark:text-green-400 text-sm">
              <CheckCircle className="w-4 h-4" />
              Real
            </span>
          )}
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
          <div>
            {comparisonError ? (
              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 text-center">
                <h3 className="text-lg font-bold text-red-600 dark:text-red-400 mb-2">
                  Error Loading Comparison Data
                </h3>
                <p className="text-red-700 dark:text-red-300 mb-4">
                  {comparisonError}
                </p>
                <button 
                  onClick={() => window.location.reload()}
                  className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
                >
                  Retry
                </button>
              </div>
            ) : loadingComparison ? (
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 text-center">
                <p className="text-blue-700 dark:text-blue-300">
                  Loading comparison data...
                </p>
              </div>
            ) : !fullApiData ? (
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 text-center">
                <h3 className="text-lg font-bold text-yellow-600 dark:text-yellow-400 mb-2">
                  No Comparison Data Available
                </h3>
                <p className="text-yellow-700 dark:text-yellow-300">
                  Please select a mitigation strategy and ensure the analysis has been completed.
                </p>
              </div>
            ) : overallFairnessScoreBefore === undefined || overallFairnessScoreAfter === undefined ? (
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 text-center">
                <h3 className="text-lg font-bold text-yellow-600 dark:text-yellow-400 mb-2">
                  Incomplete Fairness Data
                </h3>
                <p className="text-yellow-700 dark:text-yellow-300">
                  Overall fairness scores are not available in the analysis results.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                    Current Model Status
                  </h4>
                  <div className={`text-3xl font-bold ${getFairnessScoreColor(overallFairnessScoreBefore)}`}>
                    {overallFairnessScoreBefore}/100
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {getFairnessScoreLabel(overallFairnessScoreBefore)}
                  </div>
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                  <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                    After {selectedStrategy}
                  </h4>
                  <div className={`text-3xl font-bold ${getFairnessScoreColor(overallFairnessScoreAfter)}`}>
                    {overallFairnessScoreAfter}/100
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {getFairnessScoreLabel(overallFairnessScoreAfter)}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {selectedTab === 1 && (
          <div>
            {!visualizationData ? (
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 text-center">
                <h3 className="text-lg font-bold text-yellow-600 dark:text-yellow-400 mb-2">
                  No Fairness Metrics Data Available
                </h3>
                <p className="text-yellow-700 dark:text-yellow-300">
                  Please ensure the analysis has been completed and try again.
                </p>
              </div>
            ) : visualizationData.radar.before.length === 0 || visualizationData.radar.after.length === 0 ? (
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 text-center">
                <h3 className="text-lg font-bold text-yellow-600 dark:text-yellow-400 mb-2">
                  Incomplete Fairness Metrics
                </h3>
                <p className="text-yellow-700 dark:text-yellow-300">
                  Fairness metrics data is incomplete or missing from the analysis results.
                </p>
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between mb-6">
                  <p className="text-gray-700 dark:text-gray-300">
                    Fairness metrics comparison showing improvement across all dimensions:
                  </p>
                  {showRealResults && (
                    <div className="flex items-center gap-2 text-green-600 dark:text-green-400 text-sm">
                      <CheckCircle className="w-4 h-4" />
                      <span className="font-medium">Real Results</span>
                    </div>
                  )}
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Radar Chart */}
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
                    <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Visual Comparison
                    </h4>
                    <FairnessRadarChart data={visualizationData.radar} />
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
                          {visualizationData.radar.metrics.map((metric, index) => {
                            const before = visualizationData.radar.before[index];
                            const after = visualizationData.radar.after[index];
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
                                    {improvement > 0 ? '+' : ''}{improvement.toFixed(1)}
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
              </>
            )}
          </div>
        )}

        {selectedTab === 2 && (
          <div>
            <div className="flex items-center justify-between mb-6">
              <p className="text-gray-700 dark:text-gray-300">
                Performance metrics showing the trade-off between fairness and accuracy:
              </p>
              {showRealResults && (
                <div className="flex items-center gap-2 text-green-600 dark:text-green-400 text-sm">
                  <CheckCircle className="w-4 h-4" />
                  <span className="font-medium">Real Results</span>
                </div>
              )}
            </div>
            
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
                    {fullApiData?.performance ? (
                      Object.entries(fullApiData.performance.before).map(([metric, beforeValue]) => {
                        const afterValue = fullApiData.performance.after[metric];
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
                      })
                    ) : (
                      <tr>
                        <td colSpan="5" className="px-6 py-8 text-center">
                          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 mx-4">
                            <p className="text-yellow-800 dark:text-yellow-200">
                              Performance metrics not available
                            </p>
                          </div>
                        </td>
                      </tr>
                    )}
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
                  {availableAttributes.map((attr) => (
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
            
            {visualizationData?.groupComparison ? (
              <ComparisonBarChart data={visualizationData.groupComparison} />
            ) : (
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 text-center">
                <p className="text-yellow-800 dark:text-yellow-200">
                  Group comparison visualization not available
                </p>
              </div>
            )}
            
            {/* Multi-Attribute Summary */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg shadow-md p-6 mt-6">
              <h4 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                Multi-Attribute Mitigation Summary
              </h4>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                The <span className="font-bold">{selectedStrategy}</span> strategy addresses bias across all detected sensitive attributes:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {availableAttributes.map((attr) => {
                  const attrData = visualizationData?.attributeComparison?.[attr];
                  if (!attrData) {
                    return (
                      <div key={attr} className="bg-gray-50 dark:bg-gray-700 rounded-lg shadow-md p-4 h-full">
                        <h5 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                          {getAttributeDisplayName(attr)}
                        </h5>
                        <p className="text-sm text-red-600 dark:text-red-400">
                          Analysis not available
                        </p>
                      </div>
                    );
                  }
                  
                  const beforeValues = Object.values(attrData.before || {});
                  const afterValues = Object.values(attrData.after || {});
                  const groups = Object.keys(attrData.before || {});
                  
                  if (beforeValues.length === 0 || afterValues.length === 0) {
                    return (
                      <div key={attr} className="bg-gray-50 dark:bg-gray-700 rounded-lg shadow-md p-4 h-full">
                        <h5 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                          {getAttributeDisplayName(attr)}
                        </h5>
                        <p className="text-sm text-red-600 dark:text-red-400">
                          No group data available
                        </p>
                      </div>
                    );
                  }
                  
                  const maxDiffBefore = Math.max(...beforeValues) - Math.min(...beforeValues);
                  const maxDiffAfter = Math.max(...afterValues) - Math.min(...afterValues);
                  
                  // Calculate improvement percentage (reduction in disparity)
                  let improvement = 0;
                  if (maxDiffBefore > 0) {
                    improvement = ((maxDiffBefore - maxDiffAfter) / maxDiffBefore * 100);
                  }
                  
                  const improvementDisplay = improvement > 0 ? `${improvement.toFixed(1)}%` : 'No improvement';
                  const improvementColor = improvement > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
                  
                  return (
                    <div key={attr} className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 h-full">
                      <h5 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                        {getAttributeDisplayName(attr)}
                      </h5>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {groups.length} groups analyzed
                      </p>
                      <div className={`text-lg font-bold ${improvementColor}`}>
                        {improvementDisplay}
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
