// API Service Status and Testing Component
// This component allows developers to test the service switching functionality

import React, { useState, useEffect } from 'react';
import { CloudOff, Cloud, Bug, Zap, Shield, Settings } from 'lucide-react';
import FairnessAPIService from '../services';

const APIServiceStatus = () => {
  const [serviceType, setServiceType] = useState('uninitialized');
  const [healthStatus, setHealthStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [testResults, setTestResults] = useState({});

  // Initialize service and check status
  useEffect(() => {
    const initializeService = async () => {
      try {
        await FairnessAPIService.initialize();
        setServiceType(FairnessAPIService.getActiveServiceType());
      } catch (error) {
        console.error('Service initialization failed:', error);
      }
    };

    initializeService();
  }, []);

  // Test API connectivity
  const testAPIConnectivity = async () => {
    setLoading(true);
    const startTime = Date.now();
    
    try {
      const health = await FairnessAPIService.healthCheck();
      const endTime = Date.now();
      
      setHealthStatus(health);
      setTestResults({
        success: true,
        responseTime: endTime - startTime,
        timestamp: new Date().toLocaleTimeString(),
      });
    } catch (error) {
      setTestResults({
        success: false,
        error: error.message,
        timestamp: new Date().toLocaleTimeString(),
      });
    } finally {
      setLoading(false);
    }
  };

  // Force service type switching (for testing)
  const switchToMockService = () => {
    console.log('Forcing mock service usage');
    FairnessAPIService.forceMockService();
    setServiceType(FairnessAPIService.getActiveServiceType());
    setHealthStatus(null);
    setTestResults({});
  };

  const switchToRealService = () => {
    console.log('Forcing real API service usage');
    FairnessAPIService.forceRealService();
    setServiceType(FairnessAPIService.getActiveServiceType());
    setHealthStatus(null);
    setTestResults({});
  };

  const resetService = () => {
    FairnessAPIService.reset();
    setServiceType('uninitialized');
    setHealthStatus(null);
    setTestResults({});
    // Re-initialize
    setTimeout(async () => {
      await FairnessAPIService.initialize();
      setServiceType(FairnessAPIService.getActiveServiceType());
    }, 100);
  };

  const getServiceIcon = () => {
    switch (serviceType) {
      case 'real':
        return <Cloud className="text-blue-600 dark:text-blue-400" />;
      case 'mock':
        return <Bug className="text-yellow-500" />;
      default:
        return <CloudOff className="text-gray-400 dark:text-gray-500" />;
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 mb-6">
      <div className="p-6">
        <div className="flex items-center mb-4">
          <Settings className="mr-2 text-blue-600 dark:text-blue-400" />
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            API Service Status
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Current Service Status */}
          <div>
            <div className="flex items-center mb-4">
              {getServiceIcon()}
              <div className="ml-2">
                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  Active Service
                </h3>
                <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                  serviceType === 'real' 
                    ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400' 
                    : serviceType === 'mock' 
                    ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
                    : 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
                }`}>
                  {serviceType === 'real' ? 'Real API' : serviceType === 'mock' ? 'Mock Service' : 'Initializing...'}
                </span>
              </div>
            </div>

            {serviceType === 'mock' && (
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 mb-4">
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  Using mock service for development. All API calls return simulated data.
                </p>
              </div>
            )}

            {serviceType === 'real' && (
              <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3 mb-4">
                <p className="text-sm text-green-800 dark:text-green-200">
                  Connected to real backend API. All data is live.
                </p>
              </div>
            )}
          </div>

          {/* Health Check Results */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <Shield className="mr-2 text-green-600 dark:text-green-400" />
                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  Health Check
                </h3>
              </div>
              <button
                onClick={testAPIConnectivity}
                disabled={loading}
                className="inline-flex items-center px-3 py-1.5 text-xs font-medium border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? <Zap className="mr-1 w-4 h-4 animate-spin" /> : <Zap className="mr-1 w-4 h-4" />}
                Test Connection
              </button>
            </div>

            {testResults.success !== undefined && (
              <div className="mt-4">
                {testResults.success ? (
                  <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3">
                    <p className="text-sm text-green-800 dark:text-green-200">
                      ✅ Connection successful
                      <br />
                      Response time: {testResults.responseTime}ms
                      <br />
                      Last tested: {testResults.timestamp}
                    </p>
                  </div>
                ) : (
                  <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
                    <p className="text-sm text-red-800 dark:text-red-200">
                      ❌ Connection failed
                      <br />
                      Error: {testResults.error}
                      <br />
                      Last tested: {testResults.timestamp}
                    </p>
                  </div>
                )}
              </div>
            )}

            {healthStatus && (
              <div className="mt-4">
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Backend Status: {healthStatus.status} | Version: {healthStatus.version}
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-gray-200 dark:border-gray-700 my-6"></div>

        {/* Developer Controls */}
        <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
          Developer Controls
        </h3>
        
        <div className="flex flex-wrap gap-3">
          <button
            onClick={switchToMockService}
            className={`inline-flex items-center px-3 py-2 text-sm font-medium rounded-md border ${
              serviceType === 'mock' 
                ? 'bg-yellow-600 text-white border-yellow-600 hover:bg-yellow-700 dark:bg-yellow-500 dark:hover:bg-yellow-600' 
                : 'bg-white dark:bg-gray-800 text-yellow-600 dark:text-yellow-400 border-yellow-300 dark:border-yellow-600 hover:bg-yellow-50 dark:hover:bg-yellow-900/20'
            }`}
          >
            <Bug className="mr-2 w-4 h-4" />
            Use Mock Service
          </button>
          
          <button
            onClick={switchToRealService}
            className={`inline-flex items-center px-3 py-2 text-sm font-medium rounded-md border ${
              serviceType === 'real' 
                ? 'bg-blue-600 text-white border-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600' 
                : 'bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 border-blue-300 dark:border-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20'
            }`}
          >
            <Cloud className="mr-2 w-4 h-4" />
            Use Real API
          </button>
          
          <button
            onClick={resetService}
            className="inline-flex items-center px-3 py-2 text-sm font-medium border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <CloudOff className="mr-2 w-4 h-4" />
            Auto-Detect
          </button>
        </div>

        <p className="text-xs text-gray-500 dark:text-gray-400 mt-4">
          Note: Service switching is for development and testing purposes. 
          Production deployments should use environment variables.
        </p>
      </div>
    </div>
  );
};

export default APIServiceStatus;
