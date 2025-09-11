import React, { useState } from 'react';
import { Scale, BarChart3, TrendingUp } from 'lucide-react';
import SensitiveFeatureDetection from './sections/SensitiveFeatureDetection';
import FairnessMetricsDashboard from './sections/FairnessMetricsDashboard';
import MitigationStrategyAnalysis from './sections/MitigationStrategyAnalysis';
import { useAppContext } from '../context/AppContext';

const MainContent = () => {
  const [activeTab, setActiveTab] = useState(0);
  const { state } = useAppContext();

  const handleTabChange = (newValue) => {
    setActiveTab(newValue);
  };

  // Extract analysis results from global state
  const { sensitiveFeatures, mitigationStrategies } = state.analysis.results;

  // Debug: Log analysis state
  console.log('MainContent analysis state:', state.analysis);
  console.log('MainContent analysis ID:', state.analysis.id);

  const tabs = [
    {
      label: 'Sensitive Feature Detection',
      icon: <Scale className="w-5 h-5 mr-2 text-blue-600" />,
      component: <SensitiveFeatureDetection data={sensitiveFeatures} />
    },
    {
      label: 'Fairness Metrics',
      icon: <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />,
      component: <FairnessMetricsDashboard analysisId={state.analysis.id} />
    },
    {
      label: 'Mitigation Analysis',
      icon: <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />,
      component: <MitigationStrategyAnalysis data={mitigationStrategies} />
    }
  ];

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="px-6 py-4">
          <h1 className="text-3xl font-bold flex items-center text-gray-900 dark:text-gray-100 mb-2">
            <Scale className="mr-3 text-blue-600" />
            ML Fairness Evaluation Platform
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Assess and mitigate bias in your machine learning models
          </p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="px-6">
          <div className="flex space-x-8">
            {tabs.map((tab, index) => (
              <button
                key={index}
                onClick={() => handleTabChange(index)}
                className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm transition-colors duration-150 ${
                  activeTab === index
                    ? 'border-blue-600 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-auto bg-gray-50 dark:bg-gray-800">
        <div className="p-6 h-full">
          {tabs[activeTab]?.component}
        </div>
      </div>
    </div>
  );
};

export default MainContent;
