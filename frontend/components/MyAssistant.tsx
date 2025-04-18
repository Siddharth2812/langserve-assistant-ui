'use client';

// import { useEdgeRuntime } from "@assistant-ui/react";
import { useChat } from 'ai/react';
import { Thread } from '@assistant-ui/react';
import { makeMarkdownText } from '@assistant-ui/react-markdown';
import { useVercelUseChatRuntime } from '@assistant-ui/react-ai-sdk';
import { GetStockPriceToolUI } from './GetStockPriceToolUI';
import { ToolFallback } from './ToolFallBack';
import { AnalyzeDataToolUI } from './AnalyzeDataToolUI';
import { AnalysisGraphs } from './AnalysisGraphs';

const MarkdownText = makeMarkdownText();

export function MyAssistant() {
  // const runtime = useEdgeRuntime({ api: "/api/chat" });
  const chat = useChat({
    api: '/api/chat',
  });

  const runtime = useVercelUseChatRuntime(chat);

  return (
    <div className="flex h-screen">
      <div className="flex-1 flex flex-row gap-4 p-4 max-w-[1600px] mx-auto overflow-y-auto">
        <div className="flex-1 min-w-[400px]">
          <Thread
            runtime={runtime}
            assistantMessage={{ components: { Text: MarkdownText, ToolFallback } }}
            tools={[GetStockPriceToolUI, AnalyzeDataToolUI]}
          />
        </div>
        <div className="w-[600px] overflow-y-auto">
          <AnalysisGraphs />
        </div>
      </div>
    </div>
  );
}
