import React, { useState } from 'react';
import { Play } from 'lucide-react';
import FileUploader from './FileUploader';
import SensitiveAttributeSelector from './SensitiveAttributeSelector';
import { useAppContext } from '../../context/AppContext';
import { ACTION_TYPES } from '../../context/AppContext';
import FairnessAPIService from '../../services';

const Sidebar = () => {
  const { state, dispatch } = useAppContext();
  const [columns, setColumns] = useState([]);
  const [isAnalysisRunning, setIsAnalysisRunning] = useState(false);
  
  const { datasets, configuration, analysis } = state;
  const { training, testing, model } = datasets;
  const { targetColumn, sensitiveAttributes, autoDetectionEnabled } = configuration;

  const handleFileUpload = async (type, fileData) => {
    try {
      let response;
      
      // Upload file to backend using appropriate API
      if (type === 'training') {
        response = await FairnessAPIService.uploadTrainingData(fileData.file);
      } else if (type === 'testing') {
        response = await FairnessAPIService.uploadTestingData(fileData.file);
      } else if (type === 'model') {
        response = await FairnessAPIService.uploadModel(fileData.file);
      }

      // Store the response with upload ID for later use
      const uploadedData = {
        ...fileData,
        uploadId: response.upload_id,
        uploadResponse: response
      };

      dispatch({
        type: ACTION_TYPES.SET_DATASET,
        payload: { type, data: uploadedData },
      });

      // If training dataset is uploaded, extract columns for target selection
      if (type === 'training' && response.columns) {
        setColumns(response.columns);
      }
    } catch (error) {
      console.error(`Failed to upload ${type}:`, error);
      // Still set the file data for UI feedback, but mark as failed
      dispatch({
        type: ACTION_TYPES.SET_DATASET,
        payload: { 
          type, 
          data: { 
            ...fileData, 
            uploadError: error.message,
            uploadFailed: true 
          } 
        },
      });
    }
  };

  const handleTargetColumnChange = (event) => {
    dispatch({
      type: ACTION_TYPES.SET_TARGET_COLUMN,
      payload: event.target.value,
    });
  };

  const handleSensitiveAttributesChange = (attributes) => {
    dispatch({
      type: ACTION_TYPES.SET_SENSITIVE_ATTRIBUTES,
      payload: attributes,
    });
  };

  const handleAutoDetectionToggle = (event) => {
    dispatch({
      type: ACTION_TYPES.SET_AUTO_DETECTION,
      payload: event.target.checked,
    });
  };

  const handleEvaluateModel = async () => {
    setIsAnalysisRunning(true);
    
    try {
      // Validate that we have all required upload IDs
      if (!training?.uploadId || !testing?.uploadId || !model?.uploadId) {
        throw new Error('All files must be successfully uploaded before starting analysis');
      }

      // Prepare analysis configuration
      const analysisConfig = {
        training_dataset_id: training.uploadId,
        testing_dataset_id: testing.uploadId,
        model_id: model.uploadId,
        target_column: targetColumn,
        sensitive_attributes: autoDetectionEnabled ? null : sensitiveAttributes,
        auto_detection_enabled: autoDetectionEnabled,
        significance_level: 0.05
      };

      // Start analysis
      const analysisResult = await FairnessAPIService.startAnalysis(analysisConfig);
      const analysisId = analysisResult.analysisId || analysisResult.id;

      // Store analysis ID
      dispatch({
        type: ACTION_TYPES.SET_ANALYSIS_ID,
        payload: analysisId
      });

      dispatch({
        type: ACTION_TYPES.SET_ANALYSIS_RUNNING,
        payload: true
      });

      // Poll for results - check status first, then fetch results only if completed
      const pollForResults = async () => {
        try {
          // First, check analysis status
          const status = await FairnessAPIService.getAnalysisStatus(analysisId);
          
          // If analysis is not completed, just update status and continue polling
          if (status?.status !== 'completed' && !status?.isComplete) {
            dispatch({
              type: ACTION_TYPES.SET_ANALYSIS_RUNNING,
              payload: true
            });
            
            // Continue polling
            setTimeout(pollForResults, 3000);
            return;
          }

          // Analysis is completed, now fetch all results
          const [sensitiveFeatures, fairnessMetrics, mitigationStrategies] = await Promise.all([
            FairnessAPIService.getSensitiveFeatures(analysisId),
            FairnessAPIService.getFairnessMetrics(analysisId),
            FairnessAPIService.getMitigationStrategies(analysisId)
          ]);

          // Update application state with complete results
          dispatch({
            type: ACTION_TYPES.SET_ANALYSIS_RESULTS,
            payload: {
              id: analysisId,
              status,
              sensitiveFeatures,
              fairnessMetrics,
              mitigationStrategies,
              isComplete: true,
              isRunning: false
            }
          });

        } catch (error) {
          console.error('Failed to fetch analysis results:', error);
          // Fallback to mock data for demo purposes
          const { mockAPI } = await import('../../services/mockDataService');
          const [sensitiveFeatures, allFairnessMetrics, mitigationStrategies] = await Promise.all([
            mockAPI.getSensitiveFeatures(),
            mockAPI.getAllFairnessMetrics(),
            mockAPI.getMitigationStrategies()
          ]);

          dispatch({
            type: ACTION_TYPES.SET_ANALYSIS_RESULTS,
            payload: {
              id: analysisId,
              sensitiveFeatures,
              fairnessMetrics: allFairnessMetrics,
              mitigationStrategies,
              isComplete: true,
              isRunning: false,
              usingMockData: true
            }
          });
        }
      };

      // Initial poll after a short delay
      setTimeout(pollForResults, 1000);

    } catch (error) {
      console.error('Analysis failed:', error);
      
      // Fallback to mock data for demo purposes
      try {
        const { mockAPI } = await import('../../services/mockDataService');
        const [sensitiveFeatures, allFairnessMetrics, mitigationStrategies] = await Promise.all([
          mockAPI.getSensitiveFeatures(),
          mockAPI.getAllFairnessMetrics(),
          mockAPI.getMitigationStrategies()
        ]);

        dispatch({
          type: ACTION_TYPES.SET_ANALYSIS_RESULTS,
          payload: {
            sensitiveFeatures,
            fairnessMetrics: allFairnessMetrics,
            mitigationStrategies,
            isComplete: true,
            isRunning: false,
            usingMockData: true,
            error: error.message
          }
        });
      } catch (mockError) {
        console.error('Even mock data failed:', mockError);
      }
    } finally {
      setIsAnalysisRunning(false);
    }
  };

  const isEvaluationReady = training && !training.uploadFailed && training.uploadId &&
                          testing && !testing.uploadFailed && testing.uploadId &&
                          model && !model.uploadFailed && model.uploadId &&
                          targetColumn;

  return (
    <div className="p-6 h-screen overflow-y-auto bg-white dark:bg-gray-800">
      <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-6">
        Data Upload & Configuration
      </h2>

      {/* Training Dataset Upload */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Training Dataset (.csv)
        </h3>
        <FileUploader
          onFileUpload={(fileData) => handleFileUpload('training', fileData)}
          acceptedTypes=".csv"
          label="Upload Training Data"
        />
        {training && (
          <div className={`mt-2 p-3 border rounded-md ${
            training.uploadFailed 
              ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' 
              : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
          }`}>
            <p className={`text-sm ${
              training.uploadFailed 
                ? 'text-red-600 dark:text-red-400' 
                : 'text-green-600 dark:text-green-400'
            }`}>
              {training.uploadFailed 
                ? `Upload failed: ${training.uploadError}` 
                : `Training dataset uploaded: ${training.name}`}
            </p>
          </div>
        )}
      </div>

      {/* Testing Dataset Upload */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Testing Dataset (.csv)
        </h3>
        <FileUploader
          onFileUpload={(fileData) => handleFileUpload('testing', fileData)}
          acceptedTypes=".csv"
          label="Upload Testing Data"
        />
        {testing && (
          <div className={`mt-2 p-3 border rounded-md ${
            testing.uploadFailed 
              ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' 
              : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
          }`}>
            <p className={`text-sm ${
              testing.uploadFailed 
                ? 'text-red-600 dark:text-red-400' 
                : 'text-green-600 dark:text-green-400'
            }`}>
              {testing.uploadFailed 
                ? `Upload failed: ${testing.uploadError}` 
                : `Testing dataset uploaded: ${testing.name}`}
            </p>
          </div>
        )}
      </div>

      {/* Model Upload */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Model (.joblib, .pkl)
        </h3>
        <FileUploader
          onFileUpload={(fileData) => handleFileUpload('model', fileData)}
          acceptedTypes=".joblib,.pkl"
          label="Upload Trained Model"
        />
        {model && (
          <div className={`mt-2 p-3 border rounded-md ${
            model.uploadFailed 
              ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' 
              : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
          }`}>
            <p className={`text-sm ${
              model.uploadFailed 
                ? 'text-red-600 dark:text-red-400' 
                : 'text-green-600 dark:text-green-400'
            }`}>
              {model.uploadFailed 
                ? `Upload failed: ${model.uploadError}` 
                : `Model uploaded: ${model.name}`}
            </p>
          </div>
        )}
      </div>

      <hr className="my-6 border-gray-200 dark:border-gray-700" />

      {/* Target Column Selector */}
      {training && (
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Target Column
          </label>
          <select
            value={targetColumn || ''}
            onChange={handleTargetColumnChange}
            className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          >
            <option value="">Select target column...</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          {targetColumn && (
            <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
              Selected target: {targetColumn}
            </p>
          )}
        </div>
      )}

      {/* Sensitive Attribute Configuration */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Sensitive Attribute Detection
        </h3>
        
        <label className="flex items-center space-x-3 mb-4">
          <input
            type="checkbox"
            checked={autoDetectionEnabled}
            onChange={handleAutoDetectionToggle}
            className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
          />
          <span className="text-sm text-gray-700 dark:text-gray-300">
            Auto-detect sensitive attributes
          </span>
        </label>

        {!autoDetectionEnabled && (
          <div className="mt-4">
            <SensitiveAttributeSelector
              columns={columns}
              selectedAttributes={sensitiveAttributes}
              onAttributesChange={handleSensitiveAttributesChange}
            />
          </div>
        )}

        {sensitiveAttributes.length > 0 && (
          <div className="mt-4">
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
              Selected sensitive attributes:
            </p>
            <div className="flex flex-wrap gap-1">
              {sensitiveAttributes.map((attr) => (
                <span
                  key={attr}
                  className="inline-flex items-center px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 text-xs rounded-md border border-blue-300 dark:border-blue-600"
                >
                  {attr}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <hr className="my-6 border-gray-200 dark:border-gray-700" />

      {/* Analysis Action */}
      <div className="mb-6">
        <button
          onClick={handleEvaluateModel}
          disabled={!isEvaluationReady || isAnalysisRunning}
          className={`
            w-full flex items-center justify-center gap-2 py-3 px-4 rounded-md text-white font-medium transition-colors
            ${!isEvaluationReady || isAnalysisRunning 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
            }
          `}
        >
          {isAnalysisRunning ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              Analyzing Model...
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              Evaluate Model Fairness
            </>
          )}
        </button>
        
        {!isEvaluationReady && (
          <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
            Please upload datasets, model, and select target column to begin analysis
          </p>
        )}
      </div>

      {/* Analysis Status */}
      {analysis.isComplete && (
        <div className="mb-6 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md">
          <p className="text-sm text-green-600 dark:text-green-400">
            Analysis completed! Review results in the main panels.
          </p>
        </div>
      )}

      {/* Mock Data Preview */}
      <hr className="my-6 border-gray-200 dark:border-gray-700" />
      <div className="mt-6">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Demo Data Available
        </h4>
        <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
          This demo includes multi-sensitive-feature analysis for:
        </p>
        <div className="flex flex-wrap gap-1">
          {['Gender', 'Age Group', 'Race'].map((attr) => (
            <span
              key={attr}
              className="inline-flex items-center px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 text-xs rounded-md border border-gray-300 dark:border-gray-600"
            >
              {attr}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
