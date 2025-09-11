import React, { createContext, useContext, useReducer } from 'react';

// Action Types
export const ACTION_TYPES = {
  SET_DATASET: 'SET_DATASET',
  SET_TARGET_COLUMN: 'SET_TARGET_COLUMN',
  SET_SENSITIVE_ATTRIBUTES: 'SET_SENSITIVE_ATTRIBUTES',
  SET_AUTO_DETECTION: 'SET_AUTO_DETECTION',
  SET_ANALYSIS_ID: 'SET_ANALYSIS_ID',
  SET_ANALYSIS_RESULTS: 'SET_ANALYSIS_RESULTS',
  SET_ANALYSIS_RUNNING: 'SET_ANALYSIS_RUNNING',
  SET_UI_STATE: 'SET_UI_STATE',
};

// Initial State
const initialState = {
  datasets: {
    training: null,
    testing: null,
    model: null,
  },
  configuration: {
    targetColumn: null,
    sensitiveAttributes: [],
    autoDetectionEnabled: true,
  },
  analysis: {
    id: null,
    isRunning: false,
    isComplete: false,
    results: {
      sensitiveFeatures: null,
      fairnessMetrics: null,
      mitigationStrategies: null,
      beforeAfterComparisons: null,
    },
  },
  ui: {
    selectedSensitiveAttribute: 'gender',
    selectedMitigationStrategy: 'Reweighing',
    activeTab: 0,
    activeSection: 'detection',
  },
};

// Reducer
const appReducer = (state, action) => {
  switch (action.type) {
    case ACTION_TYPES.SET_DATASET:
      return {
        ...state,
        datasets: {
          ...state.datasets,
          [action.payload.type]: action.payload.data,
        },
      };

    case ACTION_TYPES.SET_TARGET_COLUMN:
      return {
        ...state,
        configuration: {
          ...state.configuration,
          targetColumn: action.payload,
        },
      };

    case ACTION_TYPES.SET_SENSITIVE_ATTRIBUTES:
      return {
        ...state,
        configuration: {
          ...state.configuration,
          sensitiveAttributes: action.payload,
        },
      };

    case ACTION_TYPES.SET_AUTO_DETECTION:
      return {
        ...state,
        configuration: {
          ...state.configuration,
          autoDetectionEnabled: action.payload,
        },
      };

    case ACTION_TYPES.SET_ANALYSIS_ID:
      return {
        ...state,
        analysis: {
          ...state.analysis,
          id: action.payload,
        },
      };

    case ACTION_TYPES.SET_ANALYSIS_RESULTS:
      return {
        ...state,
        analysis: {
          ...state.analysis,
          results: {
            ...state.analysis.results,
            ...action.payload,
          },
          isComplete: action.payload.isComplete || false,
        },
      };

    case ACTION_TYPES.SET_ANALYSIS_RUNNING:
      return {
        ...state,
        analysis: {
          ...state.analysis,
          isRunning: action.payload,
        },
      };

    case ACTION_TYPES.SET_UI_STATE:
      return {
        ...state,
        ui: {
          ...state.ui,
          ...action.payload,
        },
      };

    default:
      return state;
  }
};

// Context
const AppContext = createContext();

// Provider Component
export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const value = {
    state,
    dispatch,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

// Custom Hook
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};

export default AppContext;
