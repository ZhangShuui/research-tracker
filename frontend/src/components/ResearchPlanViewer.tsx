"use client";

import { useState } from "react";
import { ResearchPlan } from "@/lib/api";
import { MathMarkdown } from "./MathMarkdown";
import { ReviewScorePanel } from "./ReviewScorePanel";
import { RefreshCw, Send, Download, Check } from "lucide-react";
import { TranslateButton, TranslatedBlock } from "./TranslateButton";

interface Props {
  plan: ResearchPlan;
  onRefine?: (feedback: string, sections?: string[]) => void;
  isRefining?: boolean;
}

const SECTIONS = [
  { key: "introduction", label: "Introduction", num: 1 },
  { key: "related_work", label: "Related Work", num: 2 },
  { key: "methodology", label: "Methodology", num: 3 },
  { key: "experimental_design", label: "Experimental Design", num: 4 },
  { key: "expected_results", label: "Expected Results", num: 5 },
  { key: "timeline", label: "Timeline", num: 6 },
  { key: "review", label: "Peer Review", num: 7 },
] as const;

type SectionKey = (typeof SECTIONS)[number]["key"];

export function ResearchPlanViewer({ plan, onRefine, isRefining }: Props) {
  const [activeSection, setActiveSection] =
    useState<SectionKey>("introduction");
  const [showRefine, setShowRefine] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [refineAll, setRefineAll] = useState(true);

  const [translatedText, setTranslatedText] = useState<string | null>(null);

  const content = plan[activeSection] || "";
  const isReview = activeSection === "review";

  const completedSections = SECTIONS.filter((s) => plan[s.key]);
  const completedCount = completedSections.length;
  const isRunning = plan.status === "running";
  const canRefine = plan.status === "completed" && onRefine;

  // Progress percentage for the ring
  const progressPct = (completedCount / SECTIONS.length) * 100;

  function handleRefine() {
    if (!onRefine) return;
    const sections = refineAll ? undefined : [activeSection];
    onRefine(feedback.trim() || "", sections);
    setFeedback("");
    setShowRefine(false);
  }

  function handleDownloadMarkdown() {
    const md = plan.full_markdown || buildFullMarkdown();
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${plan.idea_title || "research-plan"}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function buildFullMarkdown(): string {
    const parts: string[] = [];
    parts.push(`# ${plan.idea_title || "Research Plan"}\n`);
    for (const sec of SECTIONS) {
      if (sec.key === "review") continue;
      const text = plan[sec.key];
      if (text) {
        parts.push(`## ${sec.label}\n\n${text}\n`);
      }
    }
    return parts.join("\n");
  }

  return (
    <div className="space-y-4">
      {/* Progress banner */}
      {isRunning && (
        <div className="flex items-center gap-3 text-sm text-blue-700 bg-blue-50/80 border border-blue-100 rounded-xl p-3">
          <div className="relative w-8 h-8 flex-shrink-0">
            <svg className="w-8 h-8 -rotate-90" viewBox="0 0 32 32">
              <circle
                cx="16"
                cy="16"
                r="13"
                fill="none"
                stroke="#dbeafe"
                strokeWidth="3"
              />
              <circle
                cx="16"
                cy="16"
                r="13"
                fill="none"
                stroke="#3b82f6"
                strokeWidth="3"
                strokeLinecap="round"
                strokeDasharray={`${(progressPct / 100) * 81.68} 81.68`}
                className="transition-all duration-700"
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold text-blue-600">
              {completedCount}
            </span>
          </div>
          <div>
            <p className="font-medium">Generating research plan...</p>
            <p className="text-xs text-blue-500 mt-0.5">
              {completedCount} of {SECTIONS.length} sections complete
            </p>
          </div>
        </div>
      )}

      {/* Section navigation with step indicators */}
      <div className="flex items-center gap-2">
        <div className="flex gap-1 overflow-x-auto scrollbar-hide pb-1 flex-1">
          {SECTIONS.map((sec) => {
            const hasContent = Boolean(plan[sec.key]);
            const isActive = activeSection === sec.key;
            return (
              <button
                key={sec.key}
                onClick={() => setActiveSection(sec.key)}
                disabled={!hasContent && !isActive}
                className={`group relative flex items-center gap-1.5 px-3 py-2 text-xs font-medium whitespace-nowrap rounded-lg transition-all duration-200 ${
                  isActive
                    ? "bg-slate-800 text-white shadow-sm"
                    : hasContent
                    ? "bg-slate-50 text-slate-600 hover:bg-slate-100 hover:text-slate-800"
                    : "bg-slate-50/50 text-slate-300 cursor-not-allowed"
                }`}
              >
                {/* Step indicator dot */}
                <span
                  className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold flex-shrink-0 transition-colors ${
                    isActive
                      ? "bg-white text-slate-800"
                      : hasContent
                      ? "bg-emerald-100 text-emerald-700"
                      : "bg-slate-100 text-slate-300"
                  }`}
                >
                  {hasContent && !isActive ? (
                    <Check size={9} strokeWidth={3} />
                  ) : (
                    sec.num
                  )}
                </span>
                {sec.label}
                {isRunning && !hasContent && (
                  <span className="ml-0.5 inline-block w-1.5 h-1.5 bg-slate-300 rounded-full animate-pulse" />
                )}
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-1.5 flex-shrink-0">
          {/* Download button */}
          {plan.status === "completed" && (
            <button
              onClick={handleDownloadMarkdown}
              className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 transition-colors whitespace-nowrap"
              title="Download as Markdown"
            >
              <Download size={12} />
              <span className="hidden sm:inline">Download</span>
            </button>
          )}

          {/* Translate button */}
          {content && (
            <TranslateButton
              sourceType="research_plan"
              sourceId={plan.id}
              field={activeSection}
              content={content}
              onTranslated={setTranslatedText}
            />
          )}

          {/* Refine button */}
          {canRefine && (
            <button
              onClick={() => setShowRefine(!showRefine)}
              disabled={isRefining}
              className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200 rounded-lg hover:bg-amber-100 transition-colors disabled:opacity-50 whitespace-nowrap"
            >
              <RefreshCw
                size={12}
                className={isRefining ? "animate-spin" : ""}
              />
              {isRefining ? "Refining..." : "Refine"}
            </button>
          )}
        </div>
      </div>

      {/* Refine input panel */}
      {showRefine && canRefine && (
        <div className="bg-amber-50/80 border border-amber-200 rounded-xl p-4 space-y-3">
          <p className="text-xs text-amber-700 font-medium">
            Optionally describe how to improve, or leave empty to refine based on peer review:
          </p>
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="(Optional) e.g., Address reviewer concerns, add baselines, strengthen metrics..."
            rows={3}
            className="w-full px-3 py-2 text-sm border border-amber-200 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-amber-300 bg-white"
          />
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-2 text-xs text-amber-700 cursor-pointer">
              <input
                type="checkbox"
                checked={refineAll}
                onChange={(e) => setRefineAll(e.target.checked)}
                className="rounded border-amber-300"
              />
              Refine all sections (uncheck to refine only &ldquo;
              {SECTIONS.find((s) => s.key === activeSection)?.label}&rdquo;)
            </label>
            <button
              onClick={handleRefine}
              disabled={isRefining}
              className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition-colors disabled:opacity-50"
            >
              <Send size={12} />
              {feedback.trim() ? "Refine with Feedback" : "Refine from Review"}
            </button>
          </div>
        </div>
      )}

      {/* Review score panel (shown above review content) */}
      {isReview && plan.review && <ReviewScorePanel review={plan.review} />}

      {/* Section content */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 min-h-[300px]">
        {content ? (
          <>
            <MathMarkdown className="prose prose-sm prose-slate max-w-none prose-headings:tracking-tight prose-headings:font-semibold prose-a:text-blue-600">
              {content}
            </MathMarkdown>
            {translatedText && <TranslatedBlock content={translatedText} />}
          </>
        ) : isRunning ? (
          <div className="flex flex-col items-center justify-center py-16 text-slate-400">
            <div className="w-6 h-6 border-2 border-slate-300 border-t-transparent rounded-full animate-spin mb-3" />
            <p className="text-sm">Generating this section...</p>
          </div>
        ) : (
          <p className="text-sm text-slate-400 text-center py-16">
            No content available for this section.
          </p>
        )}
      </div>

      {/* Review history (shown on review tab when history exists) */}
      {isReview && plan.review_history && plan.review_history.length > 1 && (
        <ReviewHistory history={plan.review_history} />
      )}
    </div>
  );
}

function ReviewHistory({
  history,
}: {
  history: { round: number; review: string; feedback?: string | null }[];
}) {
  const [expandedRound, setExpandedRound] = useState<number | null>(null);

  // Show all rounds except the latest (which is already displayed as the current review)
  const pastRounds = history.slice(0, -1);
  if (pastRounds.length === 0) return null;

  return (
    <div className="bg-slate-50 rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-200 bg-slate-100/50">
        <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wide">
          Review History ({history.length} rounds)
        </h4>
      </div>
      <div className="divide-y divide-slate-200">
        {pastRounds.map((entry) => (
          <div key={entry.round}>
            <button
              onClick={() =>
                setExpandedRound(
                  expandedRound === entry.round ? null : entry.round
                )
              }
              className="w-full px-4 py-2.5 text-left text-sm flex items-center justify-between hover:bg-slate-100 transition-colors"
            >
              <span className="text-slate-700 font-medium">
                Round {entry.round}
                {entry.feedback && (
                  <span className="text-slate-400 font-normal ml-2 text-xs">
                    Feedback: &ldquo;{entry.feedback.slice(0, 60)}
                    {entry.feedback.length > 60 ? "..." : ""}&rdquo;
                  </span>
                )}
                {!entry.feedback && (
                  <span className="text-slate-400 font-normal ml-2 text-xs">
                    (auto-refine from review)
                  </span>
                )}
              </span>
              <span className="text-slate-400 text-xs">
                {expandedRound === entry.round ? "Collapse" : "Expand"}
              </span>
            </button>
            {expandedRound === entry.round && entry.review && (
              <div className="px-4 pb-4">
                <div className="bg-white rounded-lg border border-slate-200 p-4">
                  <MathMarkdown className="prose prose-xs prose-slate max-w-none">
                    {entry.review}
                  </MathMarkdown>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
