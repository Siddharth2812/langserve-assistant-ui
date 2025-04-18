import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Fetch data from Django backend
    const response = await fetch('http://127.0.0.1:8001/modelv3/get_graph_data/');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Transform data to match the format we use in display_graph
    const transformedData = data.anova_data.map((item: any) => ({
      feature: item.feature_name,
      p_value: item.p_value,
      log_p_value: -Math.log10(item.p_value)
    }));

    // Sort by p-value for better visualization (matching Django's sorting)
    transformedData.sort((a: any, b: any) => a.p_value - b.p_value);

    return NextResponse.json({
      anova_data: transformedData,
      metadata: {
        total_features: data.metadata.total_features,
        significant_features: data.metadata.significant_features
      }
    });
    
  } catch (error) {
    console.error('Error fetching graph data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch graph data' },
      { status: 500 }
    );
  }
} 