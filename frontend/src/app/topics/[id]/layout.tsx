"use client";

import { useState } from "react";
import { useParams, useRouter, usePathname } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  Play,
  Square,
  Pencil,
  Trash2,
  RefreshCw,
} from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { TopicForm } from "@/components/TopicForm";

const TABS = [
  { label: "Overview", href: "" },
  { label: "Papers", href: "/papers" },
  { label: "Insights", href: "/insights" },
  { label: "Brainstorm", href: "/brainstorm" },
  { label: "Research Plan", href: "/research-plan" },
] as const;

export default function TopicLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const pathname = usePathname();
  const qc = useQueryClient();
  const [showEditForm, setShowEditForm] = useState(false);

  const { data: topic, isLoading } = useQuery({
    queryKey: ["topic", id],
    queryFn: () => api.getTopic(id),
    refetchInterval: 10_000,
  });

  const runMut = useMutation({
    mutationFn: () => api.runTopic(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["topic", id] });
      qc.invalidateQueries({ queryKey: ["sessions", id] });
    },
  });

  const stopMut = useMutation({
    mutationFn: () => api.stopTopic(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["topic", id] }),
  });

  const deleteMut = useMutation({
    mutationFn: () => api.deleteTopic(id),
    onSuccess: () => router.push("/"),
  });

  // Poll progress when running (must be before any early returns)
  const isRunning = topic?.is_running;
  const { data: progress } = useQuery({
    queryKey: ["progress", id],
    queryFn: () => api.getTopicProgress(id),
    enabled: !!isRunning,
    refetchInterval: 2_000,
  });

  function handleDelete() {
    if (!confirm(`Delete topic "${topic?.name}"? This cannot be undone.`)) return;
    deleteMut.mutate();
  }

  // Determine active tab
  const basePath = `/topics/${id}`;
  const activeTab = TABS.find(
    (t) =>
      t.href
        ? pathname === `${basePath}${t.href}`
        : pathname === basePath
  ) ?? TABS[0];

  if (isLoading) {
    return (
      <main className="max-w-5xl mx-auto px-4 py-8">
        <div className="h-8 w-48 bg-gray-200 rounded animate-pulse mb-4" />
        <div className="h-64 bg-white rounded-xl border border-gray-200 animate-pulse" />
      </main>
    );
  }

  if (!topic) {
    return (
      <main className="max-w-5xl mx-auto px-4 py-8 text-center">
        <p className="text-gray-500">Topic not found.</p>
        <Link href="/" className="text-blue-600 hover:underline text-sm mt-2 block">
          Back to dashboard
        </Link>
      </main>
    );
  }

  return (
    <main className="max-w-5xl mx-auto px-4 py-8 space-y-4">
      {/* Pipeline progress banner */}
      {isRunning && progress?.running && (
        <PipelineProgressBanner progress={progress} />
      )}

      {/* Header */}
      <div className="flex items-start gap-3">
        <Link
          href="/"
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors mt-0.5"
        >
          <ArrowLeft size={16} />
        </Link>
        <div className="flex-1 min-w-0">
          <h1 className="text-xl font-bold text-gray-900 truncate">
            {topic.name}
          </h1>
          {topic.description && (
            <p className="text-sm text-gray-500 mt-0.5">{topic.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {isRunning ? (
            <button
              onClick={() => stopMut.mutate()}
              disabled={stopMut.isPending}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors disabled:opacity-50"
            >
              <Square size={14} />
              Stop
            </button>
          ) : (
            <button
              onClick={() => runMut.mutate()}
              disabled={runMut.isPending}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {runMut.isPending ? (
                <RefreshCw size={14} className="animate-spin" />
              ) : (
                <Play size={14} />
              )}
              Run Now
            </button>
          )}
          <button
            onClick={() => setShowEditForm(true)}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            title="Edit topic"
          >
            <Pencil size={14} />
          </button>
          <button
            onClick={handleDelete}
            disabled={deleteMut.isPending}
            className="p-2 rounded-lg hover:bg-red-50 text-red-500 transition-colors"
            title="Delete topic"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {/* Tab navigation */}
      <nav className="flex gap-1 border-b border-gray-200 overflow-x-auto scrollbar-hide">
        {TABS.map((tab) => {
          const href = tab.href ? `${basePath}${tab.href}` : basePath;
          const isActive = activeTab === tab;
          return (
            <Link
              key={tab.label}
              href={href}
              className={`px-4 py-2.5 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
                isActive
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>

      {/* Tab content */}
      {children}

      {/* Edit form */}
      {showEditForm && (
        <TopicForm topic={topic} onClose={() => setShowEditForm(false)} />
      )}
    </main>
  );
}

const STAGE_LABELS: Record<string, string> = {
  starting: "Initializing",
  fetching: "Fetching sources",
  deduplicating: "Deduplicating",
  summarizing: "Summarizing papers",
  filtering: "Quality filtering",
  saving: "Saving to database",
  report: "Generating report",
  insights: "Generating insights",
};

const STAGES = ["starting", "fetching", "deduplicating", "summarizing", "filtering", "saving", "report", "insights"];

function PipelineProgressBanner({ progress }: { progress: Record<string, unknown> }) {
  const stage = (progress.stage as string) || "starting";
  const stageLabel = STAGE_LABELS[stage] || stage;
  const message = (progress.message as string) || "";

  const stageIdx = STAGES.indexOf(stage);
  let pct = stageIdx >= 0 ? Math.round((stageIdx / STAGES.length) * 100) : 0;

  if (stage === "fetching" && progress.sources_total) {
    const basePct = Math.round((1 / STAGES.length) * 100);
    const fetchPct = Math.round((((progress.sources_done as number) || 0) / (progress.sources_total as number)) * basePct);
    pct = basePct + fetchPct;
  }

  if (stage === "summarizing" && progress.papers_total) {
    const basePct = Math.round((3 / STAGES.length) * 100);
    const sumPct = Math.round((((progress.papers_done as number) || 0) / (progress.papers_total as number)) * (100 / STAGES.length));
    pct = basePct + sumPct;
  }

  // Count stats
  const papersFetched = progress.papers_fetched as number | undefined;
  const papersNew = progress.papers_new as number | undefined;
  const reposNew = progress.repos_new as number | undefined;

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-xl px-5 py-4 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <RefreshCw size={14} className="text-amber-600 animate-spin" />
          <span className="text-sm font-semibold text-amber-800">{stageLabel}</span>
          <span className="text-xs text-amber-600">({pct}%)</span>
        </div>
        <div className="flex items-center gap-3 text-xs text-amber-700">
          {papersFetched != null && papersFetched > 0 && (
            <span>{papersFetched} fetched</span>
          )}
          {papersNew != null && papersNew > 0 && (
            <span>{papersNew} papers</span>
          )}
          {reposNew != null && reposNew > 0 && (
            <span>{reposNew} repos</span>
          )}
        </div>
      </div>
      <div className="h-2 bg-amber-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-amber-400 to-amber-500 rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.max(pct, 3)}%` }}
        />
      </div>
      {message && (
        <p className="text-xs text-amber-600">{message}</p>
      )}
      {/* Stage indicators */}
      <div className="flex items-center gap-1 pt-1">
        {STAGES.map((s, i) => (
          <div key={s} className="flex items-center gap-1">
            <div
              className={`w-2 h-2 rounded-full transition-colors ${
                i < stageIdx
                  ? "bg-amber-500"
                  : i === stageIdx
                  ? "bg-amber-500 animate-pulse"
                  : "bg-amber-200"
              }`}
              title={STAGE_LABELS[s]}
            />
            {i < STAGES.length - 1 && (
              <div className={`w-3 h-px ${i < stageIdx ? "bg-amber-400" : "bg-amber-200"}`} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
