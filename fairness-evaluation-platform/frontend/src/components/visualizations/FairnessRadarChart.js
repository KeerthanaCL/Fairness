import React from 'react';
import { 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar, 
  ResponsiveContainer,
  Legend
} from 'recharts';

const FairnessRadarChart = ({ data }) => {
  if (!data || !data.before || !data.after || !data.metrics) {
    return (
      <div className="p-6 text-center">
        <p className="text-gray-500 dark:text-gray-400">No radar chart data available</p>
      </div>
    );
  }

  // Transform data for radar chart
  const chartData = data.metrics.map((metric, index) => ({
    metric: metric.length > 15 ? metric.substring(0, 15) + '...' : metric,
    fullMetric: metric,
    before: data.before[index],
    after: data.after[index],
  }));

  return (
    <div>
      <h3 className="text-lg font-bold text-center text-gray-900 dark:text-white mb-4">
        Fairness Metrics Comparison
      </h3>
      <ResponsiveContainer width="100%" height={400}>
        <RadarChart data={chartData} margin={{ top: 20, right: 80, bottom: 20, left: 80 }}>
          <PolarGrid stroke="#e0e0e0" />
          <PolarAngleAxis 
            dataKey="metric" 
            tick={{ fontSize: 12, fill: '#666' }}
          />
          <PolarRadiusAxis 
            angle={90} 
            domain={[0, 100]} 
            tick={{ fontSize: 10, fill: '#999' }}
          />
          <Radar
            name="Before Mitigation"
            dataKey="before"
            stroke="#f44336"
            fill="#f44336"
            fillOpacity={0.1}
            strokeWidth={2}
          />
          <Radar
            name="After Mitigation"
            dataKey="after"
            stroke="#4caf50"
            fill="#4caf50"
            fillOpacity={0.2}
            strokeWidth={2}
          />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>

      <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
          <span className="font-semibold">Interpretation:</span> Higher scores indicate better fairness. 
          The green area shows improvements after applying the mitigation strategy.
        </p>
      </div>
    </div>
  );
};

export default FairnessRadarChart;
