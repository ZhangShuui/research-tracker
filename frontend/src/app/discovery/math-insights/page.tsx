"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Sigma, Play, RefreshCw, Clock, Settings, ChevronDown, ChevronRight } from "lucide-react";
import { api, DiscoveryReport } from "@/lib/api";
import { DiscoveryPanel } from "@/components/DiscoveryPanel";

const CORE_CATEGORIES = [
  { id: "math.ST", label: "Statistics Theory" },
  { id: "stat.ML", label: "Machine Learning (stat)" },
  { id: "math.PR", label: "Probability" },
  { id: "math.OC", label: "Optimization & Control" },
  { id: "stat.TH", label: "Statistics Theory (stat)" },
  { id: "stat.ME", label: "Methodology" },
  { id: "math.NA", label: "Numerical Analysis" },
];

const WILDCARD_CATEGORIES = [
  { id: "math.CO", label: "Combinatorics" },
  { id: "math.AG", label: "Algebraic Geometry" },
  { id: "math.GT", label: "Geometric Topology" },
  { id: "math.DS", label: "Dynamical Systems" },
  { id: "math.FA", label: "Functional Analysis" },
  { id: "math.IT", label: "Information Theory" },
  { id: "math.LO", label: "Logic" },
  { id: "math.DG", label: "Differential Geometry" },
  { id: "math.RT", label: "Representation Theory" },
  { id: "math.CA", label: "Classical Analysis & ODEs" },
  { id: "math.AP", label: "Analysis of PDEs" },
  { id: "math.CT", label: "Category Theory" },
  { id: "physics.data-an", label: "Data Analysis (physics)" },
  { id: "nlin.CD", label: "Chaotic Dynamics" },
  { id: "q-bio.QM", label: "Quantitative Methods (bio)" },
];

const DEFAULT_CORE = CORE_CATEGORIES.map((c) => c.id);
const DEFAULT_WILDCARD = WILDCARD_CATEGORIES.map((c) => c.id);

export default function MathInsightsPage() {
  const qc = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // Settings state
  const [coreCats, setCoreCats] = useState<string[]>(DEFAULT_CORE);
  const [wildcardCats, setWildcardCats] = useState<string[]>(DEFAULT_WILDCARD);
  const [lookbackDays, setLookbackDays] = useState(14);
  const [maxRecent, setMaxRecent] = useState(100);
  const [maxHistorical, setMaxHistorical] = useState(30);
  const [maxWildcard, setMaxWildcard] = useState(15);
  const [sampleSize, setSampleSize] = useState(25);

  const { data: reportsList } = useQuery({
    queryKey: ["discovery", "math"],
    queryFn: () => api.listDiscoveryReports("math"),
    refetchInterval: 5_000,
  });

  const reports = reportsList?.reports ?? [];
  const hasRunning = reports.some((r) => r.status === "running");

  const activeReport = selectedId
    ? reports.find((r) => r.id === selectedId)
    : reports[0];

  const { data: polledReport } = useQuery({
    queryKey: ["discovery-detail", activeReport?.id],
    queryFn: () => api.getDiscoveryReport(activeReport!.id),
    enabled: !!activeReport && activeReport.status === "running",
    refetchInterval: 5_000,
  });

  const displayReport = polledReport ?? activeReport;

  const startMut = useMutation({
    mutationFn: () =>
      api.startDiscovery("math", {
        categories: coreCats,
        wildcard_categories: wildcardCats,
        lookback_days: lookbackDays,
        max_recent: maxRecent,
        max_historical: maxHistorical,
        max_wildcard: maxWildcard,
        sample_size: sampleSize,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discovery", "math"] });
    },
  });

  const reviewMut = useMutation({
    mutationFn: (reportId: string) => api.reviewDiscovery(reportId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discovery", "math"] });
    },
  });

  const regenMut = useMutation({
    mutationFn: (reportId: string) => api.regenerateDiscovery(reportId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discovery", "math"] });
    },
  });

  function toggleCat(list: string[], setList: (v: string[]) => void, id: string) {
    setList(list.includes(id) ? list.filter((c) => c !== id) : [...list, id]);
  }

  return (
    <div className="flex gap-6">
      {/* Left sidebar — settings + history */}
      <div className="w-72 flex-shrink-0 space-y-3">
        <button
          onClick={() => startMut.mutate()}
          disabled={hasRunning || startMut.isPending}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {hasRunning || startMut.isPending ? (
            <>
              <RefreshCw size={16} className="animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play size={16} />
              Discover Math
            </>
          )}
        </button>

        {/* Settings panel */}
        <div className="border rounded-xl overflow-hidden bg-white">
          <button
            type="button"
            onClick={() => setShowSettings(!showSettings)}
            className="w-full flex items-center justify-between px-3 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Settings size={14} />
              Settings
            </span>
            {showSettings ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          {showSettings && (
            <div className="px-3 pb-3 space-y-3 border-t pt-3">
              {/* Numeric params */}
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs font-medium text-gray-500">Lookback Days</label>
                  <input
                    type="number"
                    min={1}
                    max={365}
                    value={lookbackDays}
                    onChange={(e) => setLookbackDays(Number(e.target.value))}
                    className="w-full mt-0.5 px-2 py-1 text-sm border rounded-md"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-500">Sample Size</label>
                  <input
                    type="number"
                    min={5}
                    max={50}
                    value={sampleSize}
                    onChange={(e) => setSampleSize(Number(e.target.value))}
                    className="w-full mt-0.5 px-2 py-1 text-sm border rounded-md"
                  />
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="text-xs font-medium text-gray-500">Recent</label>
                  <input
                    type="number"
                    min={10}
                    max={500}
                    value={maxRecent}
                    onChange={(e) => setMaxRecent(Number(e.target.value))}
                    className="w-full mt-0.5 px-2 py-1 text-sm border rounded-md"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-500">Historical</label>
                  <input
                    type="number"
                    min={5}
                    max={100}
                    value={maxHistorical}
                    onChange={(e) => setMaxHistorical(Number(e.target.value))}
                    className="w-full mt-0.5 px-2 py-1 text-sm border rounded-md"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-500">Wildcard</label>
                  <input
                    type="number"
                    min={5}
                    max={50}
                    value={maxWildcard}
                    onChange={(e) => setMaxWildcard(Number(e.target.value))}
                    className="w-full mt-0.5 px-2 py-1 text-sm border rounded-md"
                  />
                </div>
              </div>
              <p className="text-[10px] text-gray-400 leading-tight">
                Recent/Historical/Wildcard = max papers fetched per pool. Sample Size = papers sent to LLM.
              </p>

              {/* Core categories */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-medium text-gray-500">Core Categories</label>
                  <button
                    type="button"
                    onClick={() =>
                      setCoreCats(coreCats.length === DEFAULT_CORE.length ? [] : [...DEFAULT_CORE])
                    }
                    className="text-[10px] text-purple-500 hover:text-purple-700"
                  >
                    {coreCats.length === DEFAULT_CORE.length ? "Clear" : "All"}
                  </button>
                </div>
                <div className="flex flex-wrap gap-1">
                  {CORE_CATEGORIES.map((cat) => (
                    <button
                      key={cat.id}
                      type="button"
                      onClick={() => toggleCat(coreCats, setCoreCats, cat.id)}
                      className={`px-1.5 py-0.5 text-[11px] rounded-md border transition-colors ${
                        coreCats.includes(cat.id)
                          ? "bg-purple-50 border-purple-300 text-purple-700"
                          : "bg-gray-50 border-gray-200 text-gray-400"
                      }`}
                      title={cat.id}
                    >
                      {cat.id}
                    </button>
                  ))}
                </div>
              </div>

              {/* Wildcard categories */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-medium text-gray-500">
                    Wildcard Pool <span className="text-[10px] text-gray-400">(4 sampled randomly)</span>
                  </label>
                  <button
                    type="button"
                    onClick={() =>
                      setWildcardCats(
                        wildcardCats.length === DEFAULT_WILDCARD.length ? [] : [...DEFAULT_WILDCARD]
                      )
                    }
                    className="text-[10px] text-purple-500 hover:text-purple-700"
                  >
                    {wildcardCats.length === DEFAULT_WILDCARD.length ? "Clear" : "All"}
                  </button>
                </div>
                <div className="flex flex-wrap gap-1">
                  {WILDCARD_CATEGORIES.map((cat) => (
                    <button
                      key={cat.id}
                      type="button"
                      onClick={() => toggleCat(wildcardCats, setWildcardCats, cat.id)}
                      className={`px-1.5 py-0.5 text-[11px] rounded-md border transition-colors ${
                        wildcardCats.includes(cat.id)
                          ? "bg-amber-50 border-amber-300 text-amber-700"
                          : "bg-gray-50 border-gray-200 text-gray-400"
                      }`}
                      title={cat.label}
                    >
                      {cat.id}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* History */}
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
                  displayReport?.id === r.id
                    ? "bg-purple-50 text-purple-700 border border-purple-200"
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
                  {r.started_at ? new Date(r.started_at).toLocaleDateString() : "\u2014"}
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
              <Sigma size={20} className="text-purple-500" />
              <h2 className="text-lg font-semibold text-gray-900">
                Math & Statistics Insights
              </h2>
            </div>
            <DiscoveryPanel
              report={displayReport}
              onReview={() => reviewMut.mutate(displayReport.id)}
              onRegenerate={() => regenMut.mutate(displayReport.id)}
              isReviewing={reviewMut.isPending}
              isRegenerating={regenMut.isPending}
            />
          </div>
        ) : (
          <div className="text-center py-20 text-gray-400">
            <Sigma size={48} className="mx-auto mb-4 opacity-30" />
            <p className="font-medium">No math insights yet</p>
            <p className="text-sm mt-1">
              Click &ldquo;Discover Math&rdquo; to sample and analyze recent
              papers from math and statistics arXiv categories.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
