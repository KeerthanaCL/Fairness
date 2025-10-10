import React, { useState } from 'react';
import { Search, Clock, CheckCircle, AlertCircle, Zap, Target, Grid } from 'lucide-react';

const AutoPipelineSearch = ({ analysisId, onPipelineFound }) => {
  const [searchMethod, setSearchMethod] = useState('greedy');
  const [kValue, setKValue] = useState(2);
  const [searching, setSearching] = useState(false);
  const [searchResult, setSearchResult] = useState(null);
  const [searchError, setSearchError] = useState(null);

  const searchMethods = [
    {
      id: 'greedy',
      name: 'Greedy Search',
      icon: Zap,
      description: 'Fast & practical - tests each category independently',
      time: '5-10 minutes',
      accuracy: 'Good (85-90%)',
      recommended: true
    },
    {
      id: 'topk',
      name: 'Top-K Search',
      icon: Target,
      description: 'Balanced approach - tests top combinations',
      time: '10-20 minutes',
      accuracy: 'Better (90-95%)',
      recommended: false
    },
    {
      id: 'exhaustive',
      name: 'Exhaustive Search',
      icon: Grid,
      description: 'Comprehensive - tests ALL combinations',
      time: '1-2 hours',
      accuracy: 'Best (100%)',
      recommended: false
    }
  ];

  const handleSearch = async () => {
    if (!analysisId) {
      setSearchError('Analysis ID is required');
      return;
    }

    setSearching(true);
    setSearchError(null);
    setSearchResult(null);

    try {
      const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiBaseUrl}/api/analysis/find-best-pipeline`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          analysisId: analysisId,
          searchMethod: searchMethod,
          k: searchMethod === 'topk' ? kValue : undefined
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Search failed');
      }

      const result = await response.json();
      console.log('Auto pipeline search result:', result);

      setSearchResult(result);

      // Notify parent component
      if (onPipelineFound && result.bestPipeline) {
        onPipelineFound(result.bestPipeline, result.pipelineResults);
      }

    } catch (error) {
      console.error('Auto pipeline search failed:', error);
      setSearchError(error.message || 'Failed to find best pipeline');
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
      <div className="flex items-center gap-3 mb-4">
        <Search className="w-6 h-6 text-purple-600 dark:text-purple-400" />
        <h3 className="text-xl font-bold text-gray-900 dark:text-white">
          Automatic Pipeline Search
        </h3>
      </div>

      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Let the system automatically find the best combination of mitigation strategies for your data.
      </p>

      {/* Search Method Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {searchMethods.map((method) => {
          const Icon = method.icon;
          return (
            <button
              key={method.id}
              onClick={() => setSearchMethod(method.id)}
              className={`relative p-4 rounded-lg border-2 text-left transition-all ${
                searchMethod === method.id
                  ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
              }`}
            >
              {method.recommended && (
                <span className="absolute top-2 right-2 px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400 text-xs font-medium rounded">
                  Recommended
                </span>
              )}
              
              <div className="flex items-center gap-2 mb-2">
                <Icon className={`w-5 h-5 ${searchMethod === method.id ? 'text-purple-600 dark:text-purple-400' : 'text-gray-400'}`} />
                <h4 className="font-bold text-gray-900 dark:text-white">
                  {method.name}
                </h4>
              </div>
              
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {method.description}
              </p>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500 dark:text-gray-500">Time:</span>
                  <span className="font-medium text-gray-700 dark:text-gray-300">{method.time}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500 dark:text-gray-500">Accuracy:</span>
                  <span className="font-medium text-gray-700 dark:text-gray-300">{method.accuracy}</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* K Value Selection for Top-K */}
      {searchMethod === 'topk' && (
        <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            K Value (strategies to test per category):
          </label>
          <select
            value={kValue}
            onChange={(e) => setKValue(parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value={2}>2 (8 combinations - faster)</option>
            <option value={3}>3 (27 combinations - balanced)</option>
            <option value={4}>4 (64 combinations - thorough)</option>
          </select>
        </div>
      )}

      {/* Warning for Exhaustive */}
      {searchMethod === 'exhaustive' && (
        <div className="mb-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
          <div className="flex items-center gap-2 text-yellow-800 dark:text-yellow-200 mb-2">
            <AlertCircle className="w-5 h-5" />
            <span className="font-medium">Warning: Long Processing Time</span>
          </div>
          <p className="text-sm text-yellow-700 dark:text-yellow-300">
            Exhaustive search will test all possible combinations (~75-100 pipelines). This may take 1-2 hours to complete.
          </p>
        </div>
      )}

      {/* Start Search Button */}
      <div className="flex justify-center">
        <button
          onClick={handleSearch}
          disabled={searching || !analysisId}
          className={`flex items-center gap-2 px-8 py-3 rounded-lg font-medium transition-colors ${
            searching
              ? 'bg-gray-400 text-white cursor-not-allowed'
              : 'bg-purple-600 hover:bg-purple-700 text-white shadow-lg hover:shadow-xl'
          }`}
        >
          {searching ? (
            <>
              <Clock className="w-5 h-5 animate-spin" />
              Searching for Best Pipeline...
            </>
          ) : (
            <>
              <Search className="w-5 h-5" />
              Find Best Pipeline
            </>
          )}
        </button>
      </div>

      {/* Error Display */}
      {searchError && (
        <div className="mt-6 bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span className="font-medium">Search Error:</span>
          </div>
          <p className="text-red-700 dark:text-red-300 mt-1 text-sm">{searchError}</p>
        </div>
      )}

      {/* Results Display */}
      {searchResult && (
        <div className="mt-6 bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
          <div className="flex items-center gap-2 text-green-600 dark:text-green-400 mb-4">
            <CheckCircle className="w-6 h-6" />
            <h4 className="text-lg font-bold">Best Pipeline Found!</h4>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Preprocessing</p>
              <p className="text-sm font-bold text-gray-900 dark:text-white">
                {searchResult.bestPipeline.preprocessing || 'None'}
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">In-processing</p>
              <p className="text-sm font-bold text-gray-900 dark:text-white">
                {searchResult.bestPipeline.inprocessing || 'None'}
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Post-processing</p>
              <p className="text-sm font-bold text-gray-900 dark:text-white">
                {searchResult.bestPipeline.postprocessing || 'None'}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Final Score</p>
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                {searchResult.finalScore?.toFixed(1)}/100
              </p>
            </div>
            {searchResult.improvement !== undefined && (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Improvement</p>
                <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  +{searchResult.improvement.toFixed(1)}
                </p>
              </div>
            )}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Strategies Tested</p>
              <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {searchResult.strategiesTested}
              </p>
            </div>
          </div>

          <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            The pipeline has been automatically applied. View the results in the tabs below.
          </p>
        </div>
      )}
    </div>
  );
};

export default AutoPipelineSearch;