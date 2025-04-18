import { makeAssistantToolUI } from '@assistant-ui/react'
import ReactMarkdown from 'react-markdown'

type AnalyzeDataToolArgs = {
    problem_statement: string;
}

type AnalyzeDataToolResult = string;

const LoadingSpinner = () => (
    <div className="flex justify-center items-center p-4">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
    </div>
);

const AnalysisDisplay = ({ result }: { result: AnalyzeDataToolResult }) => {
    if (!result) return null;
    
    return (
        <div className="p-6 border rounded-lg shadow-md bg-white">
            <div className="mb-6">
                <h1 className="text-2xl font-bold text-gray-900 mb-2">Analysis Results</h1>
                <div className="h-1 w-20 bg-blue-500"></div>
            </div>
            
            <div className="prose prose-blue max-w-none">
                <ReactMarkdown>{result}</ReactMarkdown>
            </div>
            
            {/* Debug output */}
            <div className="mt-4 p-4 bg-gray-100 rounded">
                <pre className="whitespace-pre-wrap text-sm">
                    {JSON.stringify(result, null, 2)}
                </pre>
            </div>
        </div>
    );
};

export const AnalyzeDataToolUI = makeAssistantToolUI<AnalyzeDataToolArgs, AnalyzeDataToolResult>({
    toolName: "analyze_data",
    render: ({ result }) => {
        // Detailed logging of the result
        console.log('Raw result:', result);
        console.log('Result type:', typeof result);
        console.log('Result structure:', {
            isNull: result === null,
            isUndefined: result === undefined,
            constructor: result?.constructor?.name
        });
        
        if (!result) return <LoadingSpinner />;
        return <AnalysisDisplay result={result} />;
    }
}); 