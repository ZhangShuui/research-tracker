"use client";

import { useState } from "react";
import {
  FileText,
  Sparkles,
  ChevronDown,
  ChevronRight,
  Settings2,
  Loader2,
  Lightbulb,
  Pen,
} from "lucide-react";
import { BrainstormIdea, BrainstormSession } from "@/lib/api";

interface Props {
  brainstormSessions: BrainstormSession[];
  onGenerate: (
    idea: Record<string, unknown>,
    brainstormSessionId?: string
  ) => void;
  isPending: boolean;
  /** Pre-filled idea from brainstorm page navigation */
  prefillIdea?: BrainstormIdea | null;
}

export function ResearchPlanToolbar({
  brainstormSessions,
  onGenerate,
  isPending,
  prefillIdea,
}: Props) {
  const [mode, setMode] = useState<"brainstorm" | "custom">(
    prefillIdea ? "brainstorm" : "brainstorm"
  );
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIdea, setSelectedIdea] = useState<{
    idea: BrainstormIdea;
    sessionId: string;
  } | null>(prefillIdea ? { idea: prefillIdea, sessionId: "" } : null);
  const [customTitle, setCustomTitle] = useState("");
  const [customProblem, setCustomProblem] = useState("");
  const [customMethod, setCustomMethod] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const completedSessions = brainstormSessions.filter(
    (s) =>
      s.status === "completed" && s.ideas_json && s.ideas_json.length > 0
  );

  const allIdeas = completedSessions.flatMap((s) =>
    (s.ideas_json || []).map((idea) => ({ idea, sessionId: s.id }))
  );

  function handleGenerate() {
    if (mode === "brainstorm" && selectedIdea) {
      onGenerate(
        selectedIdea.idea as unknown as Record<string, unknown>,
        selectedIdea.sessionId
      );
    } else if (mode === "custom" && customTitle.trim()) {
      onGenerate({
        title: customTitle,
        problem: customProblem,
        method: customMethod,
        motivation: "",
        experiment_plan: "",
      });
    }
  }

  const canGenerate =
    !isPending &&
    ((mode === "brainstorm" && selectedIdea != null) ||
      (mode === "custom" && customTitle.trim()));

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200/60 shadow-sm overflow-hidden">
      {/* Mode toggle - pill style */}
      <div className="px-4 pt-4 pb-3">
        <div className="inline-flex items-center bg-slate-100 rounded-lg p-0.5">
          <button
            onClick={() => setMode("brainstorm")}
            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-md text-xs font-medium transition-all duration-200 ${
              mode === "brainstorm"
                ? "bg-white text-slate-800 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            <Lightbulb size={13} />
            From Brainstorm
          </button>
          <button
            onClick={() => setMode("custom")}
            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-md text-xs font-medium transition-all duration-200 ${
              mode === "custom"
                ? "bg-white text-slate-800 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            <Pen size={13} />
            Custom Idea
          </button>
        </div>
      </div>

      {/* Content area */}
      <div className="px-4 pb-4 space-y-3">
        {mode === "brainstorm" ? (
          <div className="space-y-2">
            {/* Idea selector dropdown */}
            <label className="block text-xs font-medium text-slate-500 mb-1">
              Select a brainstorm idea
            </label>
            <div className="relative">
              <button
                onClick={() => setShowDropdown(!showDropdown)}
                className="w-full flex items-center justify-between px-3.5 py-2.5 border border-slate-200 rounded-lg text-sm text-left hover:border-indigo-300 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-100 transition-all bg-white"
              >
                <span
                  className={
                    selectedIdea ? "text-slate-800 font-medium" : "text-slate-400"
                  }
                >
                  {selectedIdea
                    ? selectedIdea.idea.title
                    : "Choose an idea..."}
                </span>
                <ChevronDown
                  size={14}
                  className={`text-slate-400 transition-transform ${
                    showDropdown ? "rotate-180" : ""
                  }`}
                />
              </button>

              {showDropdown && (
                <div className="absolute z-20 w-full mt-1 bg-white border border-slate-200 rounded-xl shadow-xl max-h-64 overflow-y-auto">
                  {allIdeas.length === 0 ? (
                    <div className="p-4 text-center">
                      <Sparkles size={20} className="mx-auto text-slate-300 mb-2" />
                      <p className="text-xs text-slate-400">
                        No brainstorm ideas available. Generate ideas in the
                        Brainstorm tab first.
                      </p>
                    </div>
                  ) : (
                    allIdeas.map((item, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setSelectedIdea(item);
                          setShowDropdown(false);
                        }}
                        className={`w-full text-left px-3.5 py-3 hover:bg-indigo-50/50 transition-colors border-b border-slate-50 last:border-0 ${
                          selectedIdea?.idea.title === item.idea.title
                            ? "bg-indigo-50/40"
                            : ""
                        }`}
                      >
                        <p className="text-sm font-medium text-slate-800 line-clamp-1">
                          {item.idea.title}
                        </p>
                        <p className="text-xs text-slate-500 line-clamp-1 mt-0.5">
                          {item.idea.problem}
                        </p>
                        <div className="flex gap-2 mt-1.5">
                          {item.idea.novelty_score != null && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 font-medium border border-emerald-200/60">
                              Novelty: {item.idea.novelty_score}/10
                            </span>
                          )}
                          {item.idea.feasibility_score != null && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 font-medium border border-blue-200/60">
                              Feasibility: {item.idea.feasibility_score}/10
                            </span>
                          )}
                        </div>
                      </button>
                    ))
                  )}
                </div>
              )}
            </div>

            {selectedIdea && (
              <div className="text-xs text-slate-500 bg-gradient-to-br from-slate-50 to-slate-100/50 rounded-lg p-3.5 space-y-1.5 border border-slate-200/60">
                <p>
                  <span className="font-semibold text-slate-600">Problem:</span>{" "}
                  {selectedIdea.idea.problem}
                </p>
                <p>
                  <span className="font-semibold text-slate-600">Method:</span>{" "}
                  {selectedIdea.idea.method}
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1.5">
                Research idea title
              </label>
              <input
                type="text"
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
                placeholder="e.g., Spectral normalization for temporal consistency in video diffusion"
                className="w-full px-3.5 py-2.5 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-300 transition-all"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1.5">
                Problem statement
              </label>
              <textarea
                value={customProblem}
                onChange={(e) => setCustomProblem(e.target.value)}
                placeholder="What problem does this solve?"
                rows={2}
                className="w-full px-3.5 py-2.5 text-sm border border-slate-200 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-300 transition-all"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1.5">
                Proposed method
              </label>
              <textarea
                value={customMethod}
                onChange={(e) => setCustomMethod(e.target.value)}
                placeholder="Technical approach and key innovation"
                rows={2}
                className="w-full px-3.5 py-2.5 text-sm border border-slate-200 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-300 transition-all"
              />
            </div>
          </div>
        )}

        {/* Advanced options (expandable) */}
        <div className="border-t border-slate-100 pt-3">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-600 transition-colors py-1"
          >
            <Settings2 size={12} />
            Advanced Options
            <ChevronRight
              size={12}
              className={`transition-transform ${
                showAdvanced ? "rotate-90" : ""
              }`}
            />
          </button>

          {showAdvanced && (
            <div className="mt-2 p-3 bg-slate-50/80 rounded-lg border border-slate-100 space-y-2">
              <p className="text-[10px] text-slate-400 leading-relaxed">
                Advanced generation settings will be available in a future
                update. Current generation uses default parameters optimized for
                comprehensive research plans with peer review simulation.
              </p>
            </div>
          )}
        </div>

        {/* Generate button */}
        <button
          onClick={handleGenerate}
          disabled={!canGenerate}
          className={`w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
            canGenerate
              ? "bg-gradient-to-r from-indigo-600 to-indigo-700 text-white hover:from-indigo-700 hover:to-indigo-800 active:scale-[0.98] shadow-md shadow-indigo-500/20"
              : "bg-slate-100 text-slate-400 cursor-not-allowed"
          }`}
        >
          {isPending ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Generating Plan...
            </>
          ) : (
            <>
              <FileText size={16} />
              Generate Research Plan
            </>
          )}
        </button>
      </div>
    </div>
  );
}
