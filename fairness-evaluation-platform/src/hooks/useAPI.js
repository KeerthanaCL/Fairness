// Custom React Hooks for API Integration
// These hooks provide a clean interface for components to interact with the API

import { useState, useEffect, useCallback, useRef } from 'react';
import FairnessAPIService from '../services';

// Generic API hook with loading, error, and data state management
export const useAPI = (apiFunction, dependencies = [], options = {}) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastFetchTime, setLastFetchTime] = useState(null);
  
  const abortControllerRef = useRef(null);
  const isMountedRef = useRef(true);

  const {
    immediate = true,
    retryAttempts = 0,
    retryDelay = 1000,
    cacheTimeout = 5 * 60 * 1000, // 5 minutes
    onSuccess,
    onError
  } = options;

  // Check if cached data is still valid
  const isCacheValid = useCallback(() => {
    if (!lastFetchTime || !cacheTimeout) return false;
    return Date.now() - lastFetchTime < cacheTimeout;
  }, [lastFetchTime, cacheTimeout]);

  // Execute API call with retry logic
  const execute = useCallback(async (...args) => {
    // Use cached data if available and valid
    if (data && isCacheValid()) {
      return data;
    }

    // Cancel previous request if still running
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setLoading(true);
    setError(null);

    let lastError = null;
    const maxAttempts = retryAttempts + 1;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const result = await apiFunction(...args);
        
        if (isMountedRef.current) {
          setData(result);
          setLastFetchTime(Date.now());
          setLoading(false);
          setError(null);
          
          if (onSuccess) {
            onSuccess(result);
          }
        }
        
        return result;
      } catch (err) {
        lastError = err;
        
        // Don't retry on abort or client errors
        if (err.name === 'AbortError' || (err.status >= 400 && err.status < 500)) {
          break;
        }

        // Wait before retrying
        if (attempt < maxAttempts) {
          await new Promise(resolve => 
            setTimeout(resolve, retryDelay * Math.pow(2, attempt - 1))
          );
        }
      }
    }

    if (isMountedRef.current) {
      setError(lastError);
      setLoading(false);
      
      if (onError) {
        onError(lastError);
      }
    }

    throw lastError;
  }, [apiFunction, data, isCacheValid, retryAttempts, retryDelay, onSuccess, onError]);

  // Auto-execute on mount and dependency changes
  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate, ...dependencies]);

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Manual refresh function
  const refresh = useCallback(() => {
    setLastFetchTime(null); // Invalidate cache
    return execute();
  }, [execute]);

  return {
    data,
    loading,
    error,
    execute,
    refresh,
    isStale: !isCacheValid()
  };
};

// Hook for uploading files with progress tracking
export const useFileUpload = () => {
  const [uploads, setUploads] = useState({});
  
  const uploadFile = useCallback(async (file, type, metadata = {}) => {
    const uploadId = `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Initialize upload state
    setUploads(prev => ({
      ...prev,
      [uploadId]: {
        file,
        type,
        status: 'uploading',
        progress: 0,
        error: null,
        result: null
      }
    }));

    try {
      let result;
      
      // Simulate progress for mock service
      const isUsingMock = FairnessAPIService.isUsingMockService();
      if (isUsingMock) {
        // Simulate upload progress
        for (let progress = 10; progress <= 90; progress += 10) {
          setUploads(prev => ({
            ...prev,
            [uploadId]: { ...prev[uploadId], progress }
          }));
          await new Promise(resolve => setTimeout(resolve, 200));
        }
      }

      // Perform actual upload
      switch (type) {
        case 'training':
          result = await FairnessAPIService.uploadTrainingData(file, metadata);
          break;
        case 'testing':
          result = await FairnessAPIService.uploadTestingData(file, metadata);
          break;
        case 'model':
          result = await FairnessAPIService.uploadModel(file, metadata);
          break;
        default:
          throw new Error(`Unknown upload type: ${type}`);
      }

      // Update success state
      setUploads(prev => ({
        ...prev,
        [uploadId]: {
          ...prev[uploadId],
          status: 'completed',
          progress: 100,
          result
        }
      }));

      return { uploadId, result };
    } catch (error) {
      // Update error state
      setUploads(prev => ({
        ...prev,
        [uploadId]: {
          ...prev[uploadId],
          status: 'error',
          error: error.message
        }
      }));

      throw error;
    }
  }, []);

  const removeUpload = useCallback((uploadId) => {
    setUploads(prev => {
      const { [uploadId]: removed, ...rest } = prev;
      return rest;
    });
  }, []);

  const clearUploads = useCallback(() => {
    setUploads({});
  }, []);

  return {
    uploads,
    uploadFile,
    removeUpload,
    clearUploads
  };
};

// Hook for managing analysis state with polling
export const useAnalysis = (analysisId, pollInterval = 2000) => {
  const [analysis, setAnalysis] = useState(null);
  const [isPolling, setIsPolling] = useState(false);
  const pollIntervalRef = useRef(null);

  // Start analysis
  const startAnalysis = useCallback(async (config) => {
    try {
      const result = await FairnessAPIService.startAnalysis(config);
      setAnalysis(result);
      setIsPolling(true);
      return result;
    } catch (error) {
      console.error('Failed to start analysis:', error);
      throw error;
    }
  }, []);

  // Stop analysis
  const stopAnalysis = useCallback(async () => {
    if (analysis?.analysisId) {
      try {
        await FairnessAPIService.stopAnalysis(analysis.analysisId);
        setIsPolling(false);
        setAnalysis(prev => prev ? { ...prev, status: 'stopped' } : null);
      } catch (error) {
        console.error('Failed to stop analysis:', error);
        throw error;
      }
    }
  }, [analysis?.analysisId]);

  // Poll analysis status
  const pollStatus = useCallback(async () => {
    if (!analysisId && !analysis?.analysisId) return;
    
    const id = analysisId || analysis?.analysisId;
    
    try {
      const status = await FairnessAPIService.getAnalysisStatus(id);
      setAnalysis(prev => ({ ...prev, ...status }));
      
      // Stop polling if analysis is complete or stopped
      if (status.status === 'completed' || status.status === 'stopped' || status.status === 'error') {
        setIsPolling(false);
      }
      
      return status;
    } catch (error) {
      console.error('Failed to poll analysis status:', error);
      setIsPolling(false);
      throw error;
    }
  }, [analysisId, analysis?.analysisId]);

  // Setup polling
  useEffect(() => {
    if (isPolling && pollInterval > 0) {
      pollIntervalRef.current = setInterval(pollStatus, pollInterval);
    } else {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [isPolling, pollInterval, pollStatus]);

  // Initial status fetch if analysisId provided
  useEffect(() => {
    if (analysisId && !analysis) {
      pollStatus();
    }
  }, [analysisId, analysis, pollStatus]);

  return {
    analysis,
    isPolling,
    startAnalysis,
    stopAnalysis,
    pollStatus,
    isRunning: analysis?.status === 'running',
    isCompleted: analysis?.status === 'completed',
    isStopped: analysis?.status === 'stopped',
    hasError: analysis?.status === 'error'
  };
};

// Hook for fetching sensitive features
export const useSensitiveFeatures = (analysisId, options = {}) => {
  return useAPI(
    () => FairnessAPIService.getSensitiveFeatures(analysisId),
    [analysisId],
    { immediate: !!analysisId, ...options }
  );
};

// Hook for fetching fairness metrics
export const useFairnessMetrics = (analysisId, attribute = null, options = {}) => {
  return useAPI(
    () => FairnessAPIService.getFairnessMetrics(analysisId, attribute),
    [analysisId, attribute],
    { immediate: !!analysisId, ...options }
  );
};

// Hook for fetching mitigation strategies
export const useMitigationStrategies = (analysisId, options = {}) => {
  return useAPI(
    () => FairnessAPIService.getMitigationStrategies(analysisId),
    [analysisId],
    { immediate: !!analysisId, ...options }
  );
};

// Hook for fetching before/after comparison
export const useBeforeAfterComparison = (analysisId, strategy, options = {}) => {
  return useAPI(
    () => FairnessAPIService.getBeforeAfterComparison(analysisId, strategy),
    [analysisId, strategy],
    { immediate: !!(analysisId && strategy), ...options }
  );
};

// Hook for configuration data
export const useConfiguration = () => {
  const sensitiveAttributes = useAPI(
    () => FairnessAPIService.getAvailableSensitiveAttributes(),
    [],
    { cacheTimeout: 10 * 60 * 1000 } // Cache for 10 minutes
  );

  const mitigationOptions = useAPI(
    () => FairnessAPIService.getMitigationOptions(),
    [],
    { cacheTimeout: 10 * 60 * 1000 } // Cache for 10 minutes
  );

  return {
    sensitiveAttributes,
    mitigationOptions,
    loading: sensitiveAttributes.loading || mitigationOptions.loading,
    error: sensitiveAttributes.error || mitigationOptions.error
  };
};
