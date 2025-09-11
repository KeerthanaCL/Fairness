import React, { useState } from 'react';
import {
  AlertCircle,
  CheckCircle,
  AlertTriangle,
  Info,
  TrendingUp,
  TrendingDown,
  RefreshCw,
} from 'lucide-react';
import { 
  mockFairnessMetricsByAttribute, 
  getAttributeDisplayName,
  SENSITIVE_ATTRIBUTES 
} from '../../services/mockDataService';
import { useFairnessMetrics } from '../../hooks/useAPI';

const FairnessMetricsDashboard = ({ analysisId }) => {
  const [selectedAttribute, setSelectedAttribute] = useState(SENSITIVE_ATTRIBUTES.GENDER);

  // Debug: Log the analysisId
  console.log('FairnessMetricsDashboard received analysisId:', analysisId);

  // Use API hook for dynamic data fetching
  const {
    data: apiData,
    loading,
    error,
    refresh
  } = useFairnessMetrics(analysisId, selectedAttribute, {
    retryAttempts: 2,
    cacheTimeout: 5 * 60 * 1000, // 5 minutes
  });

  // Debug: Log the API response
  console.log('FairnessMetrics API data:', apiData);
  console.log('FairnessMetrics loading:', loading);
  console.log('FairnessMetrics error:', error);

  // Fallback to mock data if no analysis ID or API data
  const useMockData = !analysisId || (!apiData && !loading);
  const attributeData = useMockData 
    ? mockFairnessMetricsByAttribute[selectedAttribute]
    : apiData?.data;

  console.log('Using mock data:', useMockData);
  console.log('Attribute data:', attributeData);
    
  const { metrics = [], overallScore, overallRating, riskClassification, primaryIssues = [] } = attributeData || {};

  // Show loading state
  if (loading && !useMockData) {
    return (
      <div className="flex justify-center items-center min-h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100 ml-4">
          Loading fairness metrics...
        </h2>
      </div>
    );
  }

  // Show error state with retry option
  if (error && !useMockData) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <p className="text-sm text-red-800 dark:text-red-200">
            Failed to load fairness metrics: {error.message}
          </p>
          <button
            onClick={refresh}
            className="inline-flex items-center px-3 py-1.5 text-sm font-medium border border-red-300 dark:border-red-600 rounded-md text-red-700 dark:text-red-300 bg-white dark:bg-red-900/20 hover:bg-red-50 dark:hover:bg-red-900/40"
          >
            <RefreshCw className="mr-1 w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  const getStatusProps = (status) => {
    switch (status.toLowerCase()) {
      case 'biased':
        return { 
          color: 'text-red-600 dark:text-red-400', 
          icon: <AlertCircle className="text-red-500" />, 
          bgColor: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' 
        };
      case 'fair':
        return { 
          color: 'text-green-600 dark:text-green-400', 
          icon: <CheckCircle className="text-green-500" />, 
          bgColor: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' 
        };
      case 'warning':
        return { 
          color: 'text-yellow-600 dark:text-yellow-400', 
          icon: <AlertTriangle className="text-yellow-500" />, 
          bgColor: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800' 
        };
      default:
        return { 
          color: 'text-gray-600 dark:text-gray-400', 
          icon: <Info className="text-gray-500" />, 
          bgColor: 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800' 
        };
    }
  };

  const getOverallScoreColor = (score) => {
    if (score >= 80) return 'text-green-600 dark:text-green-400';
    if (score >= 60) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const formatMetricValue = (value, metricName) => {
    if (metricName.includes('Ratio')) {
      return value.toFixed(2);
    }
    return (value * 100).toFixed(1) + '%';
  };

  const getMetricDescription = (metric) => {
    return metric.description || 'Fairness metric';
  };

  const getTrendIcon = (status) => {
    if (status.toLowerCase() === 'fair') return <TrendingUp className="text-green-500" />;
    return <TrendingDown className="text-red-500" />;
  };

  return (
    <div>
      {/* Service Status Indicator */}
      {useMockData && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 mb-6">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            Using mock data for demonstration. Connect to backend API for real analysis.
          </p>
        </div>
      )}
      
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Model Fairness Metrics
        </h1>
        
        {/* Sensitive Attribute Selector */}
        <div className="relative">
          <select
            value={selectedAttribute}
            onChange={(e) => setSelectedAttribute(e.target.value)}
            className="appearance-none bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md px-3 py-2 pr-8 text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            {Object.values(SENSITIVE_ATTRIBUTES).map((attr) => (
              <option key={attr} value={attr}>
                {getAttributeDisplayName(attr)}
              </option>
            ))}
          </select>
          <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
            <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>

      {/* Overall Assessment */}
      <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
          <div className="text-center">
            <div className={`text-5xl font-bold ${getOverallScoreColor(overallScore)}`}>
              {overallScore}
              <span className="text-2xl text-gray-500 dark:text-gray-400">
                /100
              </span>
            </div>
            <h3 className="text-lg font-medium text-gray-600 dark:text-gray-400 mt-2">
              Fairness Score for {getAttributeDisplayName(selectedAttribute)}
            </h3>
          </div>
          
          <div className="text-center">
            <span className={`inline-flex px-4 py-2 text-base font-medium rounded-full ${
              overallScore >= 80 
                ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                : overallScore >= 60 
                ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
                : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
            }`}>
              {overallRating}
            </span>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Overall Rating
            </p>
          </div>
          
          <div className="text-center">
            <h3 className="text-lg font-semibold text-red-600 dark:text-red-400">
              {riskClassification}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Risk Classification
            </p>
            {primaryIssues.length > 0 && (
              <div className="mt-2">
                {primaryIssues.map((issue, index) => (
                  <span
                    key={index}
                    className="inline-block px-2 py-1 text-xs font-medium border border-red-300 dark:border-red-600 rounded text-red-700 dark:text-red-300 mr-1 mb-1"
                  >
                    {issue}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-2">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Fairness Progress for {getAttributeDisplayName(selectedAttribute)}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {overallScore}%
          </p>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div 
            className={`h-2 rounded-full transition-all duration-300 ${
              overallScore >= 80 
                ? 'bg-green-500' 
                : overallScore >= 60 
                ? 'bg-yellow-500' 
                : 'bg-red-500'
            }`}
            style={{ width: `${overallScore}%` }}
          ></div>
        </div>
      </div>

      {/* Fairness Metrics Cards */}
      <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-6">
        Detailed Metrics for {getAttributeDisplayName(selectedAttribute)}
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((metric, index) => {
          const statusProps = getStatusProps(metric.status);
          
          return (
            <div 
              key={index}
              className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border-2 ${statusProps.bgColor} hover:shadow-lg transition-shadow duration-200 h-full`}
            >
              <div className="p-6">
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-2">
                      {metric.name}
                    </h3>
                    <div className="flex items-center gap-2">
                      {statusProps.icon}
                      <span className={`inline-flex px-2 py-1 text-xs font-bold rounded ${statusProps.bgColor}`}>
                        {metric.status}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-3xl font-bold ${statusProps.color}`}>
                      {formatMetricValue(metric.value, metric.name)}
                    </div>
                    {getTrendIcon(metric.status)}
                  </div>
                </div>

                <div className="text-center">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                    {getMetricDescription(metric)}
                  </p>
                  
                  {metric.groupRates && (
                    <div className="mt-4">
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                        Group Rates:
                      </p>
                      <div className="flex justify-between flex-wrap gap-2">
                        {Object.entries(metric.groupRates).map(([groupName, rate], index) => (
                          <div key={groupName} className="text-center">
                            <div className="text-sm font-bold text-gray-900 dark:text-gray-100">
                              {groupName}
                            </div>
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {(rate * 100).toFixed(1)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="mt-2">
                    <p className="text-xs text-gray-500 dark:text-gray-400 cursor-help" title={metric.tooltip || ''}>
                      Threshold: {formatMetricValue(metric.threshold, metric.name)}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary Statistics */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mt-8">
        <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-6">
          Summary for {getAttributeDisplayName(selectedAttribute)}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-red-600 dark:text-red-400">
              {metrics.filter(m => m.status.toLowerCase() === 'biased').length}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Biased Metrics
            </p>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600 dark:text-green-400">
              {metrics.filter(m => m.status.toLowerCase() === 'fair').length}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Fair Metrics
            </p>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
              {attributeData?.groups?.length || 0}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Protected Groups
            </p>
          </div>
          <div className="text-center">
            <div className={`text-3xl font-bold ${getOverallScoreColor(overallScore)}`}>
              {overallScore}%
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Fairness Score
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FairnessMetricsDashboard;
