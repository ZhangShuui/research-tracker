"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Lightbulb } from "lucide-react";
import { api } from "@/lib/api";
import { TranslateButton, TranslatedBlock } from "./TranslateButton";
import { MathMarkdown } from "./MathMarkdown";

interface Props {
  topicId: string;
}

export function InsightsPanel({ topicId }: Props) {
  const [translatedText, setTranslatedText] = useState<string | null>(null);
  const { data, isLoading } = useQuery({
    queryKey: ["insights", topicId],
    queryFn: () => api.getLatestInsights(topicId),
    refetchInterval: 30_000,
  });

  return (
    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-100 p-5">
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb size={16} className="text-blue-500" />
        <h3 className="font-semibold text-sm text-blue-900">Latest Research Insights</h3>
        <span className="ml-auto flex items-center gap-2">
          {data?.content && (
            <TranslateButton
              sourceType="insight"
              sourceId={topicId}
              field="full"
              content={data.content}
              onTranslated={setTranslatedText}
            />
          )}
          {data?.session && (
            <span className="text-xs text-blue-400 font-mono">
              {data.session.id}
            </span>
          )}
        </span>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-2 text-sm text-blue-400">
          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          Loading insights…
        </div>
      ) : data?.content ? (
        <>
          <MathMarkdown className="prose prose-sm prose-blue max-w-none text-gray-700">{data.content}</MathMarkdown>
          {translatedText && <TranslatedBlock content={translatedText} />}
        </>
      ) : (
        <p className="text-sm text-blue-400 italic">
          No insights yet. Run the pipeline to generate cross-paper insights.
        </p>
      )}
    </div>
  );
}
