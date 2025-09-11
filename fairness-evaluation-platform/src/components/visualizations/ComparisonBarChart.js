import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ComparisonBarChart = ({ data }) => {
  if (!data || !data.before || !data.after) {
    return (
      <div className="p-6 text-center">
        <p className="text-gray-500 dark:text-gray-400">No comparison data available</p>
      </div>
    );
  }

  // Transform the data for the chart
  const chartData = Object.keys(data.before).map(groupName => ({
    group: groupName,
    before: data.before[groupName],
    after: data.after[groupName],
    improvement: data.after[groupName] - data.before[groupName]
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const beforeValue = payload.find(p => p.dataKey === 'before')?.value;
      const afterValue = payload.find(p => p.dataKey === 'after')?.value;
      const improvement = afterValue - beforeValue;
      
      return (
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          <h4 className="font-semibold text-gray-900 dark:text-white">
            {label} Group
          </h4>
          <p className="text-sm text-red-600 dark:text-red-400">
            Before Mitigation: {beforeValue}%
          </p>
          <p className="text-sm text-green-600 dark:text-green-400">
            After Mitigation: {afterValue}%
          </p>
          <p className={`text-sm font-bold ${
            improvement >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
          }`}>
            Change: {improvement >= 0 ? '+' : ''}{improvement.toFixed(1)}%
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div>
      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-6">
        Group-wise Positive Prediction Rates
      </h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={chartData}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 20,
          }}
          barCategoryGap="20%"
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="group" 
            tick={{ fontSize: 12 }}
            axisLine={{ stroke: '#e0e0e0' }}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            axisLine={{ stroke: '#e0e0e0' }}
            label={{ value: 'Positive Prediction Rate (%)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Bar 
            dataKey="before" 
            fill="#f44336" 
            name="Before Mitigation"
            radius={[4, 4, 0, 0]}
          />
          <Bar 
            dataKey="after" 
            fill="#4caf50" 
            name="After Mitigation"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>

      {/* Summary Statistics */}
      <div className="mt-6 flex gap-6 flex-wrap">
        {chartData.map(group => (
          <div key={group.group} className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 flex-1 min-w-[200px]">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              {group.group} Group
            </h4>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Before:</span>
              <span className="text-sm font-bold text-gray-900 dark:text-white">{group.before}%</span>
            </div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">After:</span>
              <span className="text-sm font-bold text-gray-900 dark:text-white">{group.after}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Change:</span>
              <span className={`text-sm font-bold ${
                group.improvement >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
              }`}>
                {group.improvement >= 0 ? '+' : ''}{group.improvement.toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Fairness Assessment */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 mt-6 border-l-4 border-blue-500">
        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
          Fairness Assessment
        </h4>
        <p className="text-sm text-gray-700 dark:text-gray-300">
          The gap between groups has been reduced from{' '}
          <span className="font-bold">{Math.abs(chartData[0]?.before - chartData[1]?.before).toFixed(1)}%</span>{' '}
          to <span className="font-bold">{Math.abs(chartData[0]?.after - chartData[1]?.after).toFixed(1)}%</span>,{' '}
          representing a <span className="font-bold">
            {(((Math.abs(chartData[0]?.before - chartData[1]?.before) - Math.abs(chartData[0]?.after - chartData[1]?.after)) / Math.abs(chartData[0]?.before - chartData[1]?.before)) * 100).toFixed(1)}%
          </span> improvement in fairness.
        </p>
      </div>
    </div>
  );
};

export default ComparisonBarChart;