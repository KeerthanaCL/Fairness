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
import AutoPipelineSearch from './AutoPipelineSearch';

const MitigationStrategyAnalysis = ({ data, analysisId }) => {
  // React hooks must be called first, before any conditional logic
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedStrategy, setSelectedStrategy] = useState('');
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

  const [pipelineStrategy, setPipelineStrategy] = useState({
    preprocessing: '',
    inprocessing: '',
    postprocessing: ''
  });
  const [loadingPipeline, setLoadingPipeline] = useState(false);
  const [pipelineResults, setPipelineResults] = useState(null);
  const [pipelineError, setPipelineError] = useState(null);

  // Add pipeline application handler
  const handleApplyPipeline = async () => {
    if (!analysisId) {
      setPipelineError('Analysis ID is required');
      return;
    }

    // Check if at least one strategy is selected
    if (!pipelineStrategy.preprocessing && !pipelineStrategy.inprocessing && !pipelineStrategy.postprocessing) {
      setPipelineError('Please select at least one mitigation strategy');
      return;
    }

    setLoadingPipeline(true);
    setPipelineError(null);
    
    try {
      // console.log('Applying mitigation pipeline:', pipelineStrategy);
      // Use the FairnessAPIService if available, or direct fetch
      const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const url = `${apiBaseUrl}/api/analysis/apply-mitigation-pipeline`;

      console.log('Calling URL:', url);
      
      const response = await fetch(url, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json', 
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          analysisId: analysisId,
          preprocessingStrategy: pipelineStrategy.preprocessing || null,
          inprocessingStrategy: pipelineStrategy.inprocessing || null,
          postprocessingStrategy: pipelineStrategy.postprocessing || null
        })
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);
      
      // Read response as text first to see what we're getting
      const responseText = await response.text();
      console.log('Response text (first 500 chars):', responseText.substring(0, 500));
      
      if (!response.ok) {
        // Try to parse as JSON, but handle if it's HTML
        let errorMessage = `Server returned ${response.status}`;
        try {
          const errorData = JSON.parse(responseText);
          errorMessage = errorData.detail || errorMessage;
        } catch {
          // Response is HTML (error page)
          if (responseText.includes('<!DOCTYPE') || responseText.includes('<html')) {
            errorMessage = `API endpoint not found or server error (${response.status}). Please check if the backend server is running and the endpoint exists.`;
          } else {
            errorMessage = `Server error: ${responseText.substring(0, 200)}`;
          }
        }
        throw new Error(errorMessage);
      }
      
      let result;
      try {
        result = JSON.parse(responseText);
      } catch (parseError) {
        console.error('Failed to parse response as JSON:', parseError);
        throw new Error('Invalid response format from server');
      }
      console.log('Pipeline results:', result);
      
      setPipelineResults(result.pipeline);
      
      // Update visualization data with pipeline results
      if (result.pipeline.final) {
        // You can update the existing visualization data here
        setShowRealResults(true);
      }
      
    } catch (error) {
      console.error('Pipeline application failed:', error);
      setPipelineError(`Failed to apply pipeline: ${error.message}`);
    } finally {
      setLoadingPipeline(false);
    }
  };

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

      // Only proceed if a strategy is selected and it's not empty
      if (!selectedStrategy || selectedStrategy.trim() === '') {
        setBeforeAfterData(null);
        setFullApiData(null);
        return;
      }

      if (!selectedAttribute) {
        // Wait for attributes to load
        return;
      }

      try {
        setLoadingComparison(true);
        setComparisonError(null);
        console.log('Fetching comparison data for strategy:', selectedStrategy);
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
        
        {/* Real Mitigation Button - Top Level
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
        </div> */}
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

      {/* ADD THIS: Auto Pipeline Search Component */}
      <AutoPipelineSearch 
        analysisId={analysisId}
        onPipelineFound={(bestPipeline, pipelineResults) => {
          console.log('Best pipeline found:', bestPipeline);
          console.log('Pipeline results:', pipelineResults);
          
          // Update the pipeline strategy state with the found best pipeline
          setPipelineStrategy({
            preprocessing: bestPipeline.preprocessing || '',
            inprocessing: bestPipeline.inprocessing || '',
            postprocessing: bestPipeline.postprocessing || ''
          });
          
          // Set the pipeline results if available
          if (pipelineResults) {
            setPipelineResults(pipelineResults);
            setShowRealResults(true);
          }
        }}
      />

      {/* Pipeline Strategy Selection */}
      {/* Pipeline Strategy Selection */}
      <div className="mb-8">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
          Or Select Pipeline Manually
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Alternatively, manually select strategies from each category
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          {/* Preprocessing Selection */}
          <div>
            <label htmlFor="preprocessing-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Preprocessing
              <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">(Optional)</span>
            </label>
            <select
              id="preprocessing-select"
              value={pipelineStrategy.preprocessing}
              onChange={(e) => setPipelineStrategy({...pipelineStrategy, preprocessing: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="">-- None --</option>
              {getCategoryStrategies('Preprocessing').map(strategy => (
                <option key={strategy.name} value={strategy.name}>
                  {strategy.name} {strategy.stars ? `(${strategy.stars}★)` : ''}
                </option>
              ))}
            </select>
            {pipelineStrategy.preprocessing && (
              <div className="mt-2 flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
                <CheckCircle className="w-3 h-3" />
                <span>Selected</span>
              </div>
            )}
          </div>

          {/* In-processing Selection */}
          <div>
            <label htmlFor="inprocessing-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              In-processing
              <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">(Optional)</span>
            </label>
            <select
              id="inprocessing-select"
              value={pipelineStrategy.inprocessing}
              onChange={(e) => setPipelineStrategy({...pipelineStrategy, inprocessing: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="">-- None --</option>
              {getCategoryStrategies('In-processing').map(strategy => (
                <option key={strategy.name} value={strategy.name}>
                  {strategy.name} {strategy.stars ? `(${strategy.stars}★)` : ''}
                </option>
              ))}
            </select>
            {pipelineStrategy.inprocessing && (
              <div className="mt-2 flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
                <CheckCircle className="w-3 h-3" />
                <span>Selected</span>
              </div>
            )}
          </div>

          {/* Post-processing Selection */}
          <div>
            <label htmlFor="postprocessing-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Post-processing
              <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">(Optional)</span>
            </label>
            <select
              id="postprocessing-select"
              value={pipelineStrategy.postprocessing}
              onChange={(e) => setPipelineStrategy({...pipelineStrategy, postprocessing: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="">-- None --</option>
              {getCategoryStrategies('Post-processing').map(strategy => (
                <option key={strategy.name} value={strategy.name}>
                  {strategy.name} {strategy.stars ? `(${strategy.stars}★)` : ''}
                </option>
              ))}
            </select>
            {pipelineStrategy.postprocessing && (
              <div className="mt-2 flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
                <CheckCircle className="w-3 h-3" />
                <span>Selected</span>
              </div>
            )}
          </div>
        </div>

        {/* Pipeline Flow Visualization */}
        {(pipelineStrategy.preprocessing || pipelineStrategy.inprocessing || pipelineStrategy.postprocessing) && (
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <p className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">Selected Pipeline:</p>
            <div className="flex items-center gap-2 flex-wrap">
              {pipelineStrategy.preprocessing && (
                <>
                  <span className="px-3 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-100 rounded-full text-xs font-medium">
                    {pipelineStrategy.preprocessing}
                  </span>
                  {(pipelineStrategy.inprocessing || pipelineStrategy.postprocessing) && (
                    <span className="text-blue-600 dark:text-blue-400">→</span>
                  )}
                </>
              )}
              {pipelineStrategy.inprocessing && (
                <>
                  <span className="px-3 py-1 bg-purple-100 dark:bg-purple-800 text-purple-800 dark:text-purple-100 rounded-full text-xs font-medium">
                    {pipelineStrategy.inprocessing}
                  </span>
                  {pipelineStrategy.postprocessing && (
                    <span className="text-purple-600 dark:text-purple-400">→</span>
                  )}
                </>
              )}
              {pipelineStrategy.postprocessing && (
                <span className="px-3 py-1 bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-100 rounded-full text-xs font-medium">
                  {pipelineStrategy.postprocessing}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Apply Pipeline Button */}
        <div className="mt-6 flex justify-center">
          <button
            onClick={handleApplyPipeline}
            disabled={loadingPipeline || (!pipelineStrategy.preprocessing && !pipelineStrategy.inprocessing && !pipelineStrategy.postprocessing)}
            className={`flex items-center gap-2 px-8 py-3 rounded-lg font-medium transition-colors ${
              loadingPipeline
                ? 'bg-gray-400 text-white cursor-not-allowed'
                : (!pipelineStrategy.preprocessing && !pipelineStrategy.inprocessing && !pipelineStrategy.postprocessing)
                ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl'
            }`}
          >
            {loadingPipeline ? (
              <>
                <Clock className="w-5 h-5 animate-spin" />
                Applying Pipeline...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Apply Mitigation Pipeline
              </>
            )}
          </button>
        </div>

        {/* Pipeline Error Display */}
        {pipelineError && (
          <div className="mt-4 bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
              <AlertCircle className="w-5 h-5" />
              <span className="font-medium">Pipeline Error:</span>
            </div>
            <p className="text-red-700 dark:text-red-300 mt-1 text-sm">{pipelineError}</p>
          </div>
        )}

        {/* Pipeline Results Display */}
        {pipelineResults && (
          <div className="mt-6 bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <div className="flex items-center gap-2 text-green-600 dark:text-green-400 mb-4">
              <CheckCircle className="w-6 h-6" />
              <h4 className="text-lg font-bold">Pipeline Applied Successfully!</h4>
            </div>
            
            {/* Pipeline Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Baseline Fairness</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {pipelineResults.summary.baseline_fairness.toFixed(1)}
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Final Fairness</p>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {pipelineResults.summary.final_fairness.toFixed(1)}
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Improvement</p>
                <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  +{pipelineResults.summary.fairness_improvement.toFixed(1)}
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Stages Applied</p>
                <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {pipelineResults.summary.successful_stages}/{pipelineResults.summary.total_stages}
                </p>
              </div>
            </div>

            {/* Stage-by-Stage Results */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="text-sm font-bold text-gray-900 dark:text-white mb-3">Pipeline Stages:</h5>
              <div className="space-y-2">
                {pipelineResults.stages.map((stage, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="flex items-center gap-3">
                      <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 text-xs font-bold">
                        {index + 1}
                      </span>
                      <div>
                        <span className="text-sm font-medium text-gray-900 dark:text-white block">
                          {stage.strategy}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {stage.stage.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {stage.status === 'completed' ? (
                        <>
                          <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                          <span className="text-sm font-medium text-green-600 dark:text-green-400">
                            {stage.fairness_score?.toFixed(1)}
                          </span>
                        </>
                      ) : (
                        <>
                          <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400" />
                          <span className="text-sm text-red-600 dark:text-red-400">Failed</span>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Performance Metrics */}
            {pipelineResults.improvements && (
              <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4">
                <h5 className="text-sm font-bold text-gray-900 dark:text-white mb-3">Performance Impact:</h5>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">Accuracy</p>
                    <p className={`text-lg font-bold ${pipelineResults.improvements.accuracy >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                      {pipelineResults.improvements.accuracy >= 0 ? '+' : ''}{pipelineResults.improvements.accuracy.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">Precision</p>
                    <p className={`text-lg font-bold ${pipelineResults.improvements.precision >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                      {pipelineResults.improvements.precision >= 0 ? '+' : ''}{pipelineResults.improvements.precision.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">Recall</p>
                    <p className={`text-lg font-bold ${pipelineResults.improvements.recall >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                      {pipelineResults.improvements.recall >= 0 ? '+' : ''}{pipelineResults.improvements.recall.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">F1 Score</p>
                    <p className={`text-lg font-bold ${pipelineResults.improvements.f1 >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                      {pipelineResults.improvements.f1 >= 0 ? '+' : ''}{pipelineResults.improvements.f1.toFixed(2)}%
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Before/After Model Comparison Visualizations */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
          {pipelineResults ? (
            <>
              Pipeline Results: {[
                pipelineStrategy.preprocessing,
                pipelineStrategy.inprocessing,
                pipelineStrategy.postprocessing
              ].filter(Boolean).join(' → ')}
              <span className="ml-2 inline-flex items-center gap-1 text-green-600 dark:text-green-400 text-sm">
                <CheckCircle className="w-4 h-4" />
                Pipeline Applied
              </span>
            </>
          ) : showRealResults ? (
            <>
              Real Results: {selectedStrategy}
              <span className="ml-2 inline-flex items-center gap-1 text-green-600 dark:text-green-400 text-sm">
                <CheckCircle className="w-4 h-4" />
                Real
              </span>
            </>
          ) : (
            `Strategy Analysis: ${selectedStrategy}`
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
            {pipelineResults ? (
              // Show pipeline results in overview
              <div>
                <div className="mb-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                  <p className="text-sm text-blue-900 dark:text-blue-100">
                    <strong>Pipeline Applied:</strong>{' '}
                    {[
                      pipelineStrategy.preprocessing,
                      pipelineStrategy.inprocessing,
                      pipelineStrategy.postprocessing
                    ].filter(Boolean).join(' → ')}
                  </p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                      Baseline Model
                    </h4>
                    <div className={`text-3xl font-bold ${getFairnessScoreColor(Math.round(pipelineResults.summary.baseline_fairness))}`}>
                      {Math.round(pipelineResults.summary.baseline_fairness)}/100
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {getFairnessScoreLabel(Math.round(pipelineResults.summary.baseline_fairness))}
                    </div>
                  </div>
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                    <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-2">
                      After Pipeline
                    </h4>
                    <div className={`text-3xl font-bold ${getFairnessScoreColor(Math.round(pipelineResults.summary.final_fairness))}`}>
                      {Math.round(pipelineResults.summary.final_fairness)}/100
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {getFairnessScoreLabel(Math.round(pipelineResults.summary.final_fairness))}
                    </div>
                    <div className="mt-2 text-lg font-bold text-green-600 dark:text-green-400">
                      +{pipelineResults.summary.fairness_improvement.toFixed(1)} improvement
                    </div>
                  </div>
                </div>
              </div>
            ) : comparisonError ? (
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
            {pipelineResults ? (
              // Show pipeline fairness metrics if available
              <>
                <div className="flex items-center justify-between mb-6">
                  <p className="text-gray-700 dark:text-gray-300">
                    Pipeline fairness improvement across {pipelineResults.summary.total_stages} stage(s):
                  </p>
                  <div className="flex items-center gap-2 text-green-600 dark:text-green-400 text-sm">
                    <CheckCircle className="w-4 h-4" />
                    <span className="font-medium">Pipeline Results</span>
                  </div>
                </div>
                
                {/* Display stage-by-stage fairness progression */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 mb-4">
                  <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Fairness Progression Through Pipeline
                  </h4>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded">
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        Baseline
                      </span>
                      <span className="text-lg font-bold text-gray-600 dark:text-gray-300">
                        {pipelineResults.summary.baseline_fairness.toFixed(1)}
                      </span>
                    </div>
                    {pipelineResults.stages.map((stage, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <div>
                          <span className="text-sm font-medium text-gray-900 dark:text-white block">
                            Stage {index + 1}: {stage.strategy}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {stage.stage.replace('_', ' ')}
                          </span>
                        </div>
                        {stage.status === 'completed' ? (
                          <span className="text-lg font-bold text-green-600 dark:text-green-400">
                            {stage.fairness_score?.toFixed(1)}
                          </span>
                        ) : (
                          <span className="text-sm text-red-600 dark:text-red-400">Failed</span>
                        )}
                      </div>
                    ))}
                    <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                      <span className="text-sm font-bold text-gray-900 dark:text-white">
                        Final Result
                      </span>
                      <span className="text-xl font-bold text-green-600 dark:text-green-400">
                        {pipelineResults.summary.final_fairness.toFixed(1)}
                      </span>
                    </div>
                  </div>
                </div>
              </>
            ) : !visualizationData ? (
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
