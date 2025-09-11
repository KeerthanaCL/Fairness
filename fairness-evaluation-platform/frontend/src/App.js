import React from 'react';
import { AppProvider } from './context/AppContext';
import Sidebar from './components/sidebar/Sidebar';
import MainContent from './components/MainContent';
import APIServiceStatus from './components/APIServiceStatus';

function App() {
  return (
    <AppProvider>
      <div className="flex h-screen bg-gray-50 dark:bg-gray-800">
        {/* Sidebar */}
        <div className="flex flex-col w-80 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700">
          <Sidebar />
        </div>
        
        {/* Main Content */}
        <div className="flex-1 bg-gray-50 dark:bg-gray-800 overflow-auto">
          {/* API Service Status - Development Tool */}
          <div className="p-6">
            <APIServiceStatus />
          </div>
          <MainContent />
        </div>
      </div>
    </AppProvider>
  );
}

export default App;
