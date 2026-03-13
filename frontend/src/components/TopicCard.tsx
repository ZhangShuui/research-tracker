"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Play,
  Square,
  ChevronRight,
  FileText,
  GitFork,
  Clock,
  Calendar,
} from "lucide-react";
import Link from "next/link";
import { api, Topic } from "@/lib/api";

interface Props {
  topic: Topic;
}

// Deterministic accent colors based on topic id
const ACCENT_GRADIENTS = [
  "from-blue-500 to-indigo-600",
  "from-violet-500 to-purple-600",
  "from-emerald-500 to-teal-600",
  "from-amber-500 to-orange-600",
  "from-rose-500 to-pink-600",
  "from-cyan-500 to-blue-600",
];

function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

const STAGE_LABELS: Record<string, string> = {
  starting: "Starting...",
  fetching: "Fetching sources",
  deduplicating: "Deduplicating",
  summarizing: "Summarizing",
  filtering: "Filtering",
  saving: "Saving to DB",
  report: "Generating report",
  insights: "Generating insights",
};

export function TopicCard({ topic }: Props) {
  const qc = useQueryClient();

  const runMut = useMutation({
    mutationFn: () => api.runTopic(topic.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["topics"] }),
  });

  const stopMut = useMutation({
    mutationFn: () => api.stopTopic(topic.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["topics"] }),
  });

  const latest = topic.latest_session;
  const isRunning = topic.is_running;
  const accentIdx = hashString(topic.id) % ACCENT_GRADIENTS.length;
  const accent = ACCENT_GRADIENTS[accentIdx];

  // Poll progress when running
  const { data: progress } = useQuery({
    queryKey: ["progress", topic.id],
    queryFn: () => api.getTopicProgress(topic.id),
    enabled: !!isRunning,
    refetchInterval: 2_000,
  });

  return (
    <div className="group relative bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200/60 overflow-hidden hover:shadow-lg hover:shadow-slate-200/50 hover:border-slate-300/60 transition-all duration-300 hover:-translate-y-0.5">
      {/* Top gradient accent bar */}
      <div className={`h-1.5 bg-gradient-to-r ${accent}`} />

      <div className="p-5 flex flex-col gap-3">
        {/* Title and status */}
        <div className="flex items-start justify-between gap-2">
          <Link
            href={`/topics/${topic.id}`}
            className="font-semibold text-slate-800 hover:text-indigo-600 transition-colors leading-tight text-[15px]"
          >
            {topic.name}
          </Link>
          <StatusBadge status={isRunning ? "running" : latest?.status} />
        </div>

        {/* Description */}
        {topic.description && (
          <p className="text-sm text-slate-500 line-clamp-2 leading-relaxed">
            {topic.description}
          </p>
        )}

        {/* Progress bar when running */}
        {isRunning && progress?.running && (
          <PipelineProgress progress={progress} />
        )}

        {/* Stats row */}
        <div className="flex items-center gap-4 py-2">
          {latest ? (
            <>
              <div className="flex items-center gap-1.5">
                <FileText size={13} className="text-indigo-400" />
                <span className="text-sm font-semibold text-slate-700 tabular-nums">
                  {latest.paper_count}
                </span>
                <span className="text-xs text-slate-400">papers</span>
              </div>
              <div className="flex items-center gap-1.5">
                <GitFork size={13} className="text-emerald-400" />
                <span className="text-sm font-semibold text-slate-700 tabular-nums">
                  {latest.repo_count}
                </span>
                <span className="text-xs text-slate-400">repos</span>
              </div>
            </>
          ) : (
            <span className="text-xs text-slate-400 italic">No runs yet</span>
          )}
        </div>

        {/* Metadata */}
        <div className="text-xs text-slate-400 space-y-1">
          {latest?.started_at && (
            <div className="flex items-center gap-1.5">
              <Clock size={11} className="text-slate-300" />
              <span>
                Last run:{" "}
                {new Date(latest.started_at).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
          )}
          {topic.schedule_cron && (
            <div className="flex items-center gap-1.5">
              <Calendar size={11} className="text-slate-300" />
              <code className="bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded text-[10px] font-mono">
                {topic.schedule_cron}
              </code>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 mt-auto pt-2 border-t border-slate-100">
          {isRunning ? (
            <button
              onClick={() => stopMut.mutate()}
              disabled={stopMut.isPending}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50 font-medium"
            >
              <Square size={11} />
              Stop
            </button>
          ) : (
            <button
              onClick={() => runMut.mutate()}
              disabled={runMut.isPending}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-indigo-50 text-indigo-600 rounded-lg hover:bg-indigo-100 transition-colors disabled:opacity-50 font-medium"
            >
              <Play size={11} />
              Run Now
            </button>
          )}
          <Link
            href={`/topics/${topic.id}`}
            className="flex items-center gap-1 ml-auto text-xs text-slate-400 hover:text-indigo-600 transition-colors group/link"
          >
            View
            <ChevronRight
              size={12}
              className="group-hover/link:translate-x-0.5 transition-transform"
            />
          </Link>
        </div>
      </div>
    </div>
  );
}

function PipelineProgress({ progress }: { progress: NonNullable<Awaited<ReturnType<typeof api.getTopicProgress>>> }) {
  const stage = progress.stage || "starting";
  const stageLabel = STAGE_LABELS[stage] || stage;
  const message = progress.message || "";

  // Calculate overall progress percentage
  const STAGES = ["starting", "fetching", "deduplicating", "summarizing", "filtering", "saving", "report", "insights"];
  const stageIdx = STAGES.indexOf(stage);
  let pct = stageIdx >= 0 ? Math.round((stageIdx / STAGES.length) * 100) : 0;

  // Refine within fetching stage
  if (stage === "fetching" && progress.sources_total) {
    const basePct = Math.round((1 / STAGES.length) * 100);
    const fetchPct = Math.round(((progress.sources_done || 0) / progress.sources_total) * basePct);
    pct = basePct + fetchPct;
  }

  // Refine within summarizing stage
  if (stage === "summarizing" && progress.papers_total) {
    const basePct = Math.round((3 / STAGES.length) * 100);
    const sumPct = Math.round(((progress.papers_done || 0) / progress.papers_total) * (100 / STAGES.length));
    pct = basePct + sumPct;
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-amber-700 font-medium">{stageLabel}</span>
        <span className="text-slate-400 tabular-nums">{pct}%</span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-amber-400 to-amber-500 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${Math.max(pct, 3)}%` }}
        />
      </div>
      {message && (
        <p className="text-[10px] text-slate-400 truncate">{message}</p>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status?: string }) {
  const map: Record<string, { bg: string; dot: string; text: string }> = {
    running: {
      bg: "bg-amber-50 border-amber-200",
      dot: "bg-amber-400 animate-pulse",
      text: "text-amber-700",
    },
    completed: {
      bg: "bg-emerald-50 border-emerald-200",
      dot: "bg-emerald-400",
      text: "text-emerald-700",
    },
    failed: {
      bg: "bg-red-50 border-red-200",
      dot: "bg-red-400",
      text: "text-red-700",
    },
  };
  if (!status) return null;
  const style = map[status] ?? {
    bg: "bg-slate-50 border-slate-200",
    dot: "bg-slate-400",
    text: "text-slate-600",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-full font-medium border ${style.bg} ${style.text}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
      {status}
    </span>
  );
}
