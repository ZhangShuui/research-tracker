"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { TrendingUp, Play, RefreshCw, Clock } from "lucide-react";
import { api, DiscoveryReport } from "@/lib/api";
import { DiscoveryPanel, ThemeInfo } from "@/components/DiscoveryPanel";

export default function TrendingPage() {
  const qc = useQueryClient();
  const router = useRouter();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [creatingTopic, setCreatingTopic] = useState<string | null>(null);

  const { data: reportsList } = useQuery({
    queryKey: ["discovery", "trending"],
    queryFn: () => api.listDiscoveryReports("trending"),
    refetchInterval: 5_000,
  });

  const reports = reportsList?.reports ?? [];
  const hasRunning = reports.some((r) => r.status === "running");

  // Selected or latest report
  const activeReport = selectedId
    ? reports.find((r) => r.id === selectedId)
    : reports[0];

  // Poll active running report for updates
  const { data: polledReport } = useQuery({
    queryKey: ["discovery-detail", activeReport?.id],
    queryFn: () => api.getDiscoveryReport(activeReport!.id),
    enabled: !!activeReport && activeReport.status === "running",
    refetchInterval: 5_000,
  });

  const displayReport = polledReport ?? activeReport;

  const startMut = useMutation({
    mutationFn: () => api.startDiscovery("trending"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discovery", "trending"] });
    },
  });

  const reviewMut = useMutation({
    mutationFn: (reportId: string) => api.reviewDiscovery(reportId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discovery", "trending"] });
    },
  });

  const regenMut = useMutation({
    mutationFn: (reportId: string) => api.regenerateDiscovery(reportId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discovery", "trending"] });
    },
  });

  async function handleResearchTheme(theme: ThemeInfo) {
    setCreatingTopic(theme.title);
    try {
      // Derive arXiv keywords from the theme title + techniques
      const keywords = [
        theme.title,
        ...theme.techniques.slice(0, 5),
      ];
      // Use the trending categories since the theme came from them
      const categories = ["cs.AI", "cs.LG", "cs.CV", "cs.CL"];

      const topic = await api.createTopic({
        name: theme.title,
        description: theme.description,
        arxiv_keywords: keywords,
        arxiv_categories: categories,
        arxiv_lookback_days: 7,
      });
      router.push(`/topics/${topic.id}/brainstorm`);
    } catch (err) {
      console.error("Failed to create topic from theme:", err);
      setCreatingTopic(null);
    }
  }

  return (
    <div className="flex gap-6">
      {/* Left sidebar — history */}
      <div className="w-64 flex-shrink-0 space-y-3">
        <button
          onClick={() => startMut.mutate()}
          disabled={hasRunning || startMut.isPending}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {hasRunning || startMut.isPending ? (
            <>
              <RefreshCw size={16} className="animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play size={16} />
              Discover Trends
            </>
          )}
        </button>

        <div className="space-y-1">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-1">
            History
          </h3>
          {reports.length === 0 ? (
            <p className="text-sm text-gray-400 px-1">No reports yet</p>
          ) : (
            reports.map((r) => (
              <button
                key={r.id}
                onClick={() => setSelectedId(r.id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                  (displayReport?.id === r.id)
                    ? "bg-indigo-50 text-indigo-700 border border-indigo-200"
                    : "hover:bg-gray-50 text-gray-600"
                }`}
              >
                <div className="flex items-center gap-2">
                  {r.status === "running" ? (
                    <RefreshCw size={12} className="animate-spin text-amber-500" />
                  ) : r.status === "failed" ? (
                    <span className="w-2 h-2 bg-red-400 rounded-full" />
                  ) : r.quality_score >= 0 ? (
                    <span
                      className={`w-2 h-2 rounded-full ${
                        r.quality_score >= 80
                          ? "bg-emerald-400"
                          : r.quality_score >= 60
                          ? "bg-amber-400"
                          : "bg-red-400"
                      }`}
                    />
                  ) : (
                    <span className="w-2 h-2 bg-emerald-400 rounded-full" />
                  )}
                  <span className="truncate font-medium">
                    {r.paper_count} papers
                    {r.quality_score >= 0 && (
                      <span className="text-xs text-gray-400 ml-1">({r.quality_score})</span>
                    )}
                  </span>
                </div>
                <div className="flex items-center gap-1 mt-0.5 text-xs text-gray-400">
                  <Clock size={10} />
                  {r.started_at
                    ? new Date(r.started_at).toLocaleDateString()
                    : "—"}
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Right panel — report content */}
      <div className="flex-1 min-w-0">
        {displayReport ? (
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp size={20} className="text-indigo-500" />
              <h2 className="text-lg font-semibold text-gray-900">
                Trending Research Themes
              </h2>
            </div>
            <DiscoveryPanel
              report={displayReport}
              onReview={() => reviewMut.mutate(displayReport.id)}
              onRegenerate={() => regenMut.mutate(displayReport.id)}
              onResearchTheme={handleResearchTheme}
              isReviewing={reviewMut.isPending}
              isRegenerating={regenMut.isPending}
            />
          </div>
        ) : (
          <div className="text-center py-20 text-gray-400">
            <TrendingUp size={48} className="mx-auto mb-4 opacity-30" />
            <p className="font-medium">No discovery reports yet</p>
            <p className="text-sm mt-1">
              Click &ldquo;Discover Trends&rdquo; to analyze recent papers from
              arXiv, HuggingFace, and Papers With Code.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
