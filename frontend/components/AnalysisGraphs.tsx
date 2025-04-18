import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';

type FeatureData = {
  feature: string;
  p_value: number;
  log_p_value: number;
};

type GraphData = {
  anova_data: FeatureData[];
  metadata: {
    total_features: number;
    significant_features: number;
  };
};

const FEATURE_COLORS = {
  'Air temperature [K]': '#1f77b4',
  'Process temperature [K]': '#2ca02c',
  'Rotational speed [rpm]': '#ff7f0e',
  'Tool wear [min]': '#d62728',
  'Torque [Nm]': '#9467bd'
};

export function AnalysisGraphs() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const response = await fetch('/api/graph-data');
        if (!response.ok) {
          throw new Error('Failed to fetch graph data');
        }
        const data = await response.json();
        setGraphData(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        console.error('Error fetching graph data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchGraphData();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500 p-4">
        Error loading graphs: {error}
      </div>
    );
  }

  if (!graphData?.anova_data?.length) {
    return (
      <div className="text-gray-500 p-4">
        No graph data available
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* ANOVA Results Graph */}
      {/* <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-bold mb-4">ANOVA Results: Feature Significance</h2>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart 
              data={graphData.anova_data}
              margin={{ top: 20, right: 30, left: 60, bottom: 5 }}
            >
              <CartesianGrid 
                strokeDasharray="3 3"
                horizontal={true}
                vertical={false}
              />
              <XAxis 
                dataKey="feature" 
                hide={true}
              />
              <YAxis 
                domain={[0, 100]}
                label={{ 
                  value: '-log10(p-value)', 
                  angle: -90, 
                  position: 'insideLeft',
                  offset: 10
                }}
                axisLine={{ stroke: '#E5E7EB' }}
                tick={{ fill: '#374151' }}
              />
              <Tooltip 
                formatter={(value: number) => [value.toFixed(4), '-log10(p-value)']}
                labelFormatter={(label) => `Feature: ${label}`}
                contentStyle={{ backgroundColor: 'white', border: '1px solid #E5E7EB' }}
              />
              <Bar 
                dataKey="log_p_value" 
                fill="#1f77b4"
                name="Feature Significance"
                background={{ fill: '#fff' }}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div> */}

      {/* Feature Values Graph */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-bold mb-4">Feature Values Distribution</h2>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart 
              data={graphData.anova_data}
              margin={{ top: 20, right: 30, left: 60, bottom: 5 }}
            >
              <CartesianGrid 
                strokeDasharray="3 3"
                horizontal={true}
                vertical={false}
              />
              <XAxis 
                dataKey="feature" 
                hide={true}
              />
              <YAxis 
                domain={[0, 100]}
                label={{ 
                  value: '-log10(p-value)', 
                  angle: -90, 
                  position: 'insideLeft',
                  offset: 10
                }}
                axisLine={{ stroke: '#E5E7EB' }}
                tick={{ fill: '#374151' }}
              />
              <Tooltip 
                formatter={(value: number) => [value.toFixed(4), '-log10(p-value)']}
                labelFormatter={(label) => `Feature: ${label}`}
                contentStyle={{ backgroundColor: 'white', border: '1px solid #E5E7EB' }}
              />
              <Bar 
                dataKey="log_p_value" 
                name="Feature Significance"
                background={{ fill: '#fff' }}
              >
                {graphData.anova_data.map((entry) => (
                  <Cell 
                    key={entry.feature}
                    fill={FEATURE_COLORS[entry.feature as keyof typeof FEATURE_COLORS] || '#1f77b4'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
} 