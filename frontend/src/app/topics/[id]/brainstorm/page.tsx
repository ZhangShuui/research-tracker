"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Sparkles,
  Send,
  Code2,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Settings2,
} from "lucide-react";
import { api, BrainstormSession, ContextOptions } from "@/lib/api";
import { IdeaCard } from "@/components/IdeaCard";
import { VerificationResults } from "@/components/VerificationResults";
import { BrainstormSessionCard } from "@/components/BrainstormSessionCard";

const CONTEXT_TOGGLES: {
  key: keyof ContextOptions;
  label: string;
  description: string;
}[] = [
  {
    key: "use_insights",
    label: "Topic Insights",
    description: "Research gaps & opportunities from insights report",
  },
  {
    key: "use_reports",
    label: "Session Reports",
    description: "Thematic analysis from prior research runs",
  },
  {
    key: "use_github",
    label: "GitHub Repos",
    description: "Implementation signals and engineering pain points",
  },
  {
    key: "use_history",
    label: "Brainstorm History",
    description: "Learn from past ideas — avoid repeats, build on failures",
  },
  {
    key: "use_research_plans",
    label: "Research Plans",
    description: "Already-explored research directions",
  },
  {
    key: "use_citations",
    label: "Citation Weighting",
    description: "Prioritize high-impact papers in context",
  },
  {
    key: "use_questions",
    label: "Question-First",
    description: "Generate research questions before ideas",
  },
  {
    key: "use_novelty_map",
    label: "Novelty Map",
    description: "Multi-axis novelty analysis (problem/method/eval/theory/setting)",
  },
];

const STORAGE_KEY = "brainstorm-context-options";

function loadContextOptions(): ContextOptions {
  if (typeof window === "undefined") return {};
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return JSON.parse(stored);
  } catch {}
  // Default: all on
  const defaults: ContextOptions = {};
  for (const t of CONTEXT_TOGGLES) defaults[t.key] = true;
  return defaults;
}

function saveContextOptions(opts: ContextOptions) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(opts));
  } catch {}
}

export default function BrainstormPage() {
  const { id } = useParams<{ id: string }>();
  const qc = useQueryClient();

  const [userIdea, setUserIdea] = useState("");
  const [codeVerify, setCodeVerify] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showContext, setShowContext] = useState(false);
  const [contextOpts, setContextOpts] = useState<ContextOptions>(loadContextOptions);

  // Persist to localStorage
  useEffect(() => {
    saveContextOptions(contextOpts);
  }, [contextOpts]);

  const enabledCount = Object.values(contextOpts).filter(Boolean).length;

  const { data: sessionsData } = useQuery({
    queryKey: ["brainstorm-sessions", id],
    queryFn: () => api.listBrainstormSessions(id),
    refetchInterval: 3_000,
  });

  const sessions = sessionsData?.sessions ?? [];
  const selected = sessions.find((s) => s.id === selectedId) ?? sessions[0] ?? null;
  const hasRunning = sessions.some((s) => s.status === "running");

  // Fetch full session details for selected
  const { data: detail } = useQuery({
    queryKey: ["brainstorm-detail", id, selected?.id],
    queryFn: () => api.getBrainstormSession(id, selected!.id),
    enabled: Boolean(selected),
    refetchInterval: selected?.status === "running" ? 3_000 : false,
  });

  // Poll progress when a session is running
  const runningSession = sessions.find((s) => s.status === "running");
  const { data: progress } = useQuery({
    queryKey: ["brainstorm-progress", id, runningSession?.id],
    queryFn: () => api.getBrainstormProgress(id, runningSession!.id),
    enabled: !!runningSession,
    refetchInterval: 2_000,
  });

  const autoMut = useMutation({
    mutationFn: () =>
      api.startBrainstorm(id, {
        mode: "auto",
        run_code_verification: codeVerify,
        context_options: contextOpts,
      }),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["brainstorm-sessions", id] });
      setSelectedId(data.session_id);
    },
  });

  const userMut = useMutation({
    mutationFn: () =>
      api.startBrainstorm(id, {
        mode: "user",
        user_idea: userIdea,
        run_code_verification: codeVerify,
        context_options: contextOpts,
      }),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["brainstorm-sessions", id] });
      setSelectedId(data.session_id);
      setUserIdea("");
    },
  });

  const isPending = autoMut.isPending || userMut.isPending || hasRunning;

  function toggleContext(key: keyof ContextOptions) {
    setContextOpts((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function toggleAll(on: boolean) {
    const next: ContextOptions = {};
    for (const t of CONTEXT_TOGGLES) next[t.key] = on;
    setContextOpts(next);
  }

  return (
    <div className="space-y-6">
      {/* Toolbar */}
      <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-3">
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            onClick={() => autoMut.mutate()}
            disabled={isPending}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all disabled:opacity-50 text-sm font-medium"
          >
            <Sparkles size={16} />
            {hasRunning ? "Generating..." : "Auto-Generate Ideas"}
          </button>

          <div className="flex-1 flex gap-2">
            <input
              type="text"
              value={userIdea}
              onChange={(e) => setUserIdea(e.target.value)}
              placeholder="Or describe your research idea..."
              className="input flex-1"
              onKeyDown={(e) => {
                if (e.key === "Enter" && userIdea.trim()) userMut.mutate();
              }}
            />
            <button
              onClick={() => userMut.mutate()}
              disabled={isPending || !userIdea.trim()}
              className="flex items-center gap-1.5 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 text-sm"
            >
              <Send size={14} />
              Evaluate
            </button>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-xs text-gray-500 cursor-pointer">
            <input
              type="checkbox"
              checked={codeVerify}
              onChange={(e) => setCodeVerify(e.target.checked)}
              className="rounded border-gray-300"
            />
            <Code2 size={12} />
            Code proof-of-concept
          </label>

          {/* Context sources toggle */}
          <button
            type="button"
            onClick={() => setShowContext(!showContext)}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-indigo-600 transition-colors ml-auto"
          >
            <Settings2 size={12} />
            Context Sources
            <span className="px-1.5 py-0.5 bg-indigo-50 text-indigo-600 rounded-full text-[10px] font-semibold tabular-nums">
              {enabledCount}/{CONTEXT_TOGGLES.length}
            </span>
            {showContext ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </button>
        </div>

        {/* Context Sources Panel */}
        {showContext && (
          <div className="bg-slate-50 rounded-lg border border-slate-200 p-3 space-y-2">
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs font-medium text-slate-600">
                Select which context sources to include in brainstorming
              </p>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => toggleAll(true)}
                  className="text-[10px] text-indigo-600 hover:text-indigo-800 font-medium"
                >
                  All On
                </button>
                <button
                  type="button"
                  onClick={() => toggleAll(false)}
                  className="text-[10px] text-gray-400 hover:text-gray-600 font-medium"
                >
                  All Off
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
              {CONTEXT_TOGGLES.map((t) => (
                <label
                  key={t.key}
                  className={`flex items-start gap-2 p-2 rounded-lg cursor-pointer transition-colors ${
                    contextOpts[t.key]
                      ? "bg-indigo-50/80 border border-indigo-200/60"
                      : "bg-white border border-slate-200/60 opacity-60"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={contextOpts[t.key] ?? true}
                    onChange={() => toggleContext(t.key)}
                    className="rounded border-slate-300 mt-0.5"
                  />
                  <div className="min-w-0">
                    <span className="text-xs font-medium text-slate-700 block">
                      {t.label}
                    </span>
                    <span className="text-[10px] text-slate-400 leading-tight block">
                      {t.description}
                    </span>
                  </div>
                </label>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Session history sidebar */}
        <div className="lg:col-span-1 space-y-2">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            Brainstorm History
          </h3>
          {sessions.length === 0 ? (
            <p className="text-sm text-gray-400 py-4 text-center">
              No brainstorm sessions yet.
            </p>
          ) : (
            sessions.map((s) => (
              <BrainstormSessionCard
                key={s.id}
                session={s}
                onClick={() => setSelectedId(s.id)}
                isSelected={selected?.id === s.id}
              />
            ))
          )}
        </div>

        {/* Results panel */}
        <div className="lg:col-span-2 space-y-4">
          {!detail ? (
            <div className="text-center py-16">
              <Sparkles size={32} className="mx-auto text-gray-300 mb-3" />
              <p className="text-gray-500 text-sm">
                Generate ideas to explore research directions.
              </p>
              <p className="text-gray-400 text-xs mt-1">
                Auto-generate analyzes your paper library for gaps and opportunities,
                or submit your own idea for evaluation.
              </p>
            </div>
          ) : (
            <>
              {detail.status === "running" && (
                <BrainstormProgress progress={progress} />
              )}

              {/* Ideas */}
              {detail.ideas_json && detail.ideas_json.length > 0 && (() => {
                const active = detail.ideas_json.filter((i) => i.status !== "dropped");
                const dropped = detail.ideas_json.filter((i) => i.status === "dropped");
                return (
                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-gray-700">
                      Generated Ideas ({active.length})
                      {dropped.length > 0 && (
                        <span className="text-gray-400 font-normal ml-1">
                          + {dropped.length} dropped
                        </span>
                      )}
                    </h3>
                    {active.map((idea, i) => (
                      <IdeaCard key={i} idea={idea} index={i} brainstormSessionId={detail.id} />
                    ))}
                    {dropped.length > 0 && (
                      <details className="group">
                        <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600 py-1">
                          Show {dropped.length} dropped idea{dropped.length > 1 ? "s" : ""}
                        </summary>
                        <div className="space-y-3 mt-2">
                          {dropped.map((idea, i) => (
                            <IdeaCard
                              key={`d-${i}`}
                              idea={idea}
                              index={active.length + i}
                              brainstormSessionId={detail.id}
                            />
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                );
              })()}

              {/* Verification results */}
              <VerificationResults
                literatureResult={detail.literature_result || ""}
                logicResult={detail.logic_result || ""}
                codeResult={detail.code_result || ""}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

const BRAINSTORM_STAGES = [
  "context",
  "questions",
  "research",
  "generating",
  "prescreen",
  "novelty",
  "reviewing",
  "rescue",
  "polish",
  "literature",
  "logic",
  "code",
] as const;

const BRAINSTORM_STAGE_LABELS: Record<string, string> = {
  context: "Loading Context",
  questions: "Research Questions",
  research: "Research Analysis",
  generating: "Generating Ideas",
  prescreen: "Novelty Prescreen",
  novelty: "Novelty Challenge",
  reviewing: "Review & Refine",
  rescue: "Rescuing Ideas",
  polish: "Final Polish",
  literature: "Literature Check",
  logic: "Logic Check",
  code: "Code Verification",
};

function BrainstormProgress({
  progress,
}: {
  progress?: {
    running: boolean;
    stage?: string;
    message?: string;
    ideas_count?: number;
    round?: number;
    total_rounds?: number;
    accepted?: number;
  };
}) {
  const stage = progress?.stage || "context";
  const stageLabel = BRAINSTORM_STAGE_LABELS[stage] || stage;
  const message = progress?.message || "Brainstorming in progress...";
  const stageIdx = BRAINSTORM_STAGES.indexOf(stage as (typeof BRAINSTORM_STAGES)[number]);
  const pct = stageIdx >= 0 ? Math.round((stageIdx / BRAINSTORM_STAGES.length) * 100) : 0;

  return (
    <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-5 py-4 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <RefreshCw size={14} className="text-indigo-600 animate-spin" />
          <span className="text-sm font-semibold text-indigo-800">{stageLabel}</span>
          <span className="text-xs text-indigo-500">({pct}%)</span>
        </div>
        <div className="flex items-center gap-3 text-xs text-indigo-600">
          {progress?.ideas_count != null && progress.ideas_count > 0 && (
            <span>{progress.ideas_count} ideas</span>
          )}
          {progress?.accepted != null && progress.accepted > 0 && (
            <span>{progress.accepted} accepted</span>
          )}
          {progress?.round != null && progress?.total_rounds != null && (
            <span>
              round {progress.round}/{progress.total_rounds}
            </span>
          )}
        </div>
      </div>
      <div className="h-2 bg-indigo-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-indigo-400 to-purple-500 rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.max(pct, 3)}%` }}
        />
      </div>
      {message && <p className="text-xs text-indigo-500">{message}</p>}
      {/* Stage dots */}
      <div className="flex items-center gap-1 pt-1">
        {BRAINSTORM_STAGES.map((s, i) => (
          <div key={s} className="flex items-center gap-1">
            <div
              className={`w-2 h-2 rounded-full transition-colors ${
                i < stageIdx
                  ? "bg-indigo-500"
                  : i === stageIdx
                  ? "bg-indigo-500 animate-pulse"
                  : "bg-indigo-200"
              }`}
              title={BRAINSTORM_STAGE_LABELS[s]}
            />
            {i < BRAINSTORM_STAGES.length - 1 && (
              <div
                className={`w-3 h-px ${i < stageIdx ? "bg-indigo-400" : "bg-indigo-200"}`}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
