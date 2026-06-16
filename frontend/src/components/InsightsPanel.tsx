"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Lightbulb, History, ChevronDown } from "lucide-react";
import { api, Session } from "@/lib/api";
import { TranslateButton, TranslatedBlock } from "./TranslateButton";
import { MathMarkdown } from "./MathMarkdown";

interface Props {
  topicId: string;
  /** Optional controlled selection (lifted to the page so the generate panel
   *  can jump the dropdown to a freshly created memo). When omitted, the panel
   *  manages its own selection. */
  selectedId?: string | null;
  onSelect?: (id: string) => void;
}

export function InsightsPanel({ topicId, selectedId: selectedIdProp, onSelect }: Props) {
  const [translatedText, setTranslatedText] = useState<string | null>(null);
  const [internalId, setInternalId] = useState<string | null>(null);

  // Controlled when a parent passes onSelect; otherwise self-managed.
  const controlled = typeof onSelect === "function";
  const selectedId = controlled ? selectedIdProp ?? null : internalId;
  const selectSession = (id: string) => {
    if (controlled) onSelect!(id);
    else setInternalId(id);
    setTranslatedText(null);
  };

  // List of sessions → those that actually have insights (newest first).
  const { data: sessionsData, isLoading: sessionsLoading } = useQuery({
    queryKey: ["sessions", topicId],
    queryFn: () => api.getSessions(topicId, 100),
    refetchInterval: 30_000, // pick up insights from freshly finished runs
  });

  const insightSessions = (sessionsData?.sessions ?? []).filter(
    (s) => s.insights_path
  );

  // Default to the latest session that has insights; honor an explicit pick.
  const effectiveId = selectedId ?? insightSessions[0]?.id ?? null;

  // Insights content for the selected session.
  const { data: session, isLoading: insightsLoading } = useQuery({
    queryKey: ["session-insights", topicId, effectiveId],
    queryFn: () => api.getSession(topicId, effectiveId as string),
    enabled: !!effectiveId,
  });

  const content = session?.insights_content ?? "";

  const optionLabel = (s: Session, i: number) =>
    `${s.id} · ${s.paper_count} papers` +
    (s.kind === "manual" ? " · manual" : "") +
    (s.status === "partial" ? " (partial)" : "") +
    (i === 0 ? " — latest" : "");

  return (
    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-100 p-5">
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <Lightbulb size={16} className="text-blue-500" />
        <h3 className="font-semibold text-sm text-blue-900">Research Insights</h3>
        {content && (
          <span className="ml-auto">
            <TranslateButton
              sourceType="insight"
              sourceId={topicId}
              field="full"
              content={content}
              onTranslated={setTranslatedText}
            />
          </span>
        )}
      </div>

      {/* Prominent run/memo switcher */}
      {insightSessions.length > 0 && (
        <div className="flex items-center gap-2 mb-4 flex-wrap">
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-700">
            <History size={14} />
            Viewing
          </span>
          <div className="relative">
            <select
              value={effectiveId ?? ""}
              onChange={(e) => selectSession(e.target.value)}
              className="appearance-none text-sm font-medium rounded-lg border-2 border-blue-300 bg-white pl-3 pr-9 py-1.5 text-blue-900 hover:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-400 cursor-pointer shadow-sm max-w-[300px]"
              title="Switch between insight memos (runs + manual)"
            >
              {insightSessions.map((s, i) => (
                <option key={s.id} value={s.id}>
                  {optionLabel(s, i)}
                </option>
              ))}
            </select>
            <ChevronDown
              size={16}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-blue-500 pointer-events-none"
            />
          </div>
          <span className="text-xs text-blue-400">
            {insightSessions.length} {insightSessions.length === 1 ? "memo" : "memos"}
          </span>
        </div>
      )}

      {sessionsLoading || (effectiveId && insightsLoading) ? (
        <div className="flex items-center gap-2 text-sm text-blue-400">
          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          Loading insights…
        </div>
      ) : content ? (
        <>
          <MathMarkdown className="prose prose-sm prose-blue max-w-none text-gray-700">
            {content}
          </MathMarkdown>
          {translatedText && <TranslatedBlock content={translatedText} />}
        </>
      ) : (
        <p className="text-sm text-blue-400 italic">
          No insights yet. Run the pipeline or generate insights from selected papers above.
        </p>
      )}
    </div>
  );
}
