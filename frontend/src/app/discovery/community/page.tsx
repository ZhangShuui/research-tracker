"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  MessageCircle,
  Play,
  RefreshCw,
  Clock,
  Settings,
  ChevronDown,
  ChevronRight,
  ExternalLink,
} from "lucide-react";
import { api, DiscoveryReport } from "@/lib/api";

const DEFAULT_KEYWORDS = [
  "machine learning research idea",
  "deep learning breakthrough",
  "LLM limitations practical",
  "AI research direction",
  "reinforcement learning application",
];

const PLATFORM_OPTIONS = [
  { id: "hackernews", label: "HackerNews", color: "orange" },
  { id: "reddit", label: "Reddit", color: "blue" },
  { id: "web", label: "Web/Blogs", color: "green" },
];

export default function CommunityPage() {
  const qc = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // Settings state
  const [keywords, setKeywords] = useState(DEFAULT_KEYWORDS.join("\n"));
  const [platforms, setPlatforms] = useState<string[]>(["hackernews", "reddit", "web"]);
  const [maxPerPlatform, setMaxPerPlatform] = useState(15);

  const { data: reportsList } = useQuery({
    queryKey: ["discovery", "community"],
    queryFn: () => api.listDiscoveryReports("community"),
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
    mutationFn: () => {
      const kws = keywords
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean);
      return api.startDiscovery("community", {
        keywords: kws.length > 0 ? kws : undefined,
        platforms,
        max_results_per_platform: maxPerPlatform,
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discovery", "community"] });
    },
  });

  const reviewMut = useMutation({
    mutationFn: (reportId: string) => api.reviewDiscovery(reportId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discovery", "community"] });
    },
  });

  const regenMut = useMutation({
    mutationFn: (reportId: string) => api.regenerateDiscovery(reportId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discovery", "community"] });
    },
  });

  function togglePlatform(id: string) {
    setPlatforms((prev) =>
      prev.includes(id) ? prev.filter((p) => p !== id) : [...prev, id]
    );
  }

  return (
    <div className="flex gap-6">
      {/* Left sidebar */}
      <div className="w-72 flex-shrink-0 space-y-3">
        <button
          onClick={() => startMut.mutate()}
          disabled={hasRunning || startMut.isPending || platforms.length === 0}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {hasRunning || startMut.isPending ? (
            <>
              <RefreshCw size={16} className="animate-spin" />
              Searching...
            </>
          ) : (
            <>
              <Play size={16} />
              Discover Ideas
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
            {showSettings ? (
              <ChevronDown size={14} />
            ) : (
              <ChevronRight size={14} />
            )}
          </button>
          {showSettings && (
            <div className="px-3 pb-3 space-y-3 border-t pt-3">
              {/* Platforms */}
              <div>
                <label className="text-xs font-medium text-gray-500 mb-1 block">
                  Platforms
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {PLATFORM_OPTIONS.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => togglePlatform(p.id)}
                      className={`px-2.5 py-1 text-xs rounded-lg border transition-colors ${
                        platforms.includes(p.id)
                          ? p.color === "orange"
                            ? "bg-orange-50 border-orange-300 text-orange-700"
                            : p.color === "blue"
                            ? "bg-blue-50 border-blue-300 text-blue-700"
                            : "bg-emerald-50 border-emerald-300 text-emerald-700"
                          : "bg-gray-50 border-gray-200 text-gray-400"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Max per platform */}
              <div>
                <label className="text-xs font-medium text-gray-500">
                  Results per Platform per Keyword
                </label>
                <input
                  type="number"
                  min={5}
                  max={30}
                  value={maxPerPlatform}
                  onChange={(e) => setMaxPerPlatform(Number(e.target.value))}
                  className="w-full mt-0.5 px-2 py-1 text-sm border rounded-md"
                />
              </div>

              {/* Keywords */}
              <div>
                <label className="text-xs font-medium text-gray-500">
                  Search Keywords (one per line)
                </label>
                <textarea
                  value={keywords}
                  onChange={(e) => setKeywords(e.target.value)}
                  className="w-full mt-0.5 px-2 py-1.5 text-sm border rounded-md h-32 resize-none font-mono"
                  placeholder="machine learning research idea&#10;deep learning breakthrough&#10;..."
                />
                <p className="text-[10px] text-gray-400 mt-0.5">
                  Each keyword is searched across all selected platforms.
                </p>
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
                    ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                    : "hover:bg-gray-50 text-gray-600"
                }`}
              >
                <div className="flex items-center gap-2">
                  {r.status === "running" ? (
                    <RefreshCw
                      size={12}
                      className="animate-spin text-amber-500"
                    />
                  ) : r.status === "failed" ? (
                    <span className="w-2 h-2 bg-red-400 rounded-full" />
                  ) : (
                    <span className="w-2 h-2 bg-emerald-400 rounded-full" />
                  )}
                  <span className="truncate font-medium">
                    {r.paper_count} posts
                    {Object.keys(r.source_stats || {}).length > 0 && (
                      <span className="text-xs text-gray-400 ml-1">
                        ({Object.entries(r.source_stats)
                          .map(([k, v]) => `${k}:${v}`)
                          .join(", ")})
                      </span>
                    )}
                  </span>
                </div>
                <div className="flex items-center gap-1 mt-0.5 text-xs text-gray-400">
                  <Clock size={10} />
                  {r.started_at
                    ? new Date(r.started_at).toLocaleDateString()
                    : "\u2014"}
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Right panel — report content */}
      <div className="flex-1 min-w-0">
        {displayReport ? (
          <CommunityReportView
            report={displayReport}
            onReview={() => reviewMut.mutate(displayReport.id)}
            onRegenerate={() => regenMut.mutate(displayReport.id)}
            isReviewing={reviewMut.isPending}
            isRegenerating={regenMut.isPending}
          />
        ) : (
          <div className="text-center py-20 text-gray-400">
            <MessageCircle size={48} className="mx-auto mb-4 opacity-30" />
            <p className="font-medium">No community reports yet</p>
            <p className="text-sm mt-1">
              Click &ldquo;Discover Ideas&rdquo; to search HackerNews, Reddit,
              and blogs for research-worthy discussions.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Community Report Viewer (inline, since DiscoveryPanel is theme/math specific)
// ---------------------------------------------------------------------------

function CommunityReportView({
  report,
  onReview,
  onRegenerate,
  isReviewing,
  isRegenerating,
}: {
  report: DiscoveryReport;
  onReview: () => void;
  onRegenerate: () => void;
  isReviewing: boolean;
  isRegenerating: boolean;
}) {
  if (report.status === "running") {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
        <RefreshCw size={32} className="mx-auto mb-3 text-emerald-500 animate-spin" />
        <p className="font-medium text-gray-700">
          Searching community forums...
        </p>
        <p className="text-sm text-gray-400 mt-1">
          {report.paper_count > 0
            ? `Found ${report.paper_count} discussions so far`
            : "Collecting discussions from HackerNews, Reddit, and web"}
        </p>
      </div>
    );
  }

  if (report.status === "failed") {
    return (
      <div className="bg-white rounded-xl border border-red-200 p-6">
        <p className="text-red-600 font-medium">Discovery failed</p>
        <p className="text-sm text-gray-500 mt-1">{report.content}</p>
        <button
          onClick={onRegenerate}
          disabled={isRegenerating}
          className="mt-3 px-3 py-1.5 text-sm bg-red-50 text-red-600 rounded-lg hover:bg-red-100"
        >
          Retry
        </button>
      </div>
    );
  }

  // Parse markdown content into sections
  const content = report.content || "";
  const sections = parseCommunitySections(content);

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageCircle size={20} className="text-emerald-500" />
          <h2 className="text-lg font-semibold text-gray-900">
            Community Research Ideas
          </h2>
        </div>
        <div className="flex items-center gap-2">
          {report.quality_score >= 0 && (
            <span
              className={`px-2 py-0.5 text-xs rounded-full font-medium ${
                report.quality_score >= 80
                  ? "bg-emerald-100 text-emerald-700"
                  : report.quality_score >= 60
                  ? "bg-amber-100 text-amber-700"
                  : "bg-red-100 text-red-700"
              }`}
            >
              Quality: {report.quality_score}
            </span>
          )}
          <button
            onClick={onReview}
            disabled={isReviewing}
            className="px-3 py-1.5 text-xs bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 disabled:opacity-50"
          >
            {isReviewing ? "Reviewing..." : "Review"}
          </button>
          <button
            onClick={onRegenerate}
            disabled={isRegenerating}
            className="px-3 py-1.5 text-xs bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 disabled:opacity-50"
          >
            {isRegenerating ? "Regenerating..." : "Regenerate"}
          </button>
        </div>
      </div>

      {/* Stats bar */}
      <div className="flex gap-4 text-sm text-gray-500">
        <span>{report.paper_count} discussions analyzed</span>
        {Object.entries(report.source_stats || {}).map(([k, v]) => (
          <span key={k} className="px-2 py-0.5 bg-gray-50 rounded text-xs">
            {k}: {v as number}
          </span>
        ))}
      </div>

      {/* Ideas */}
      {sections.ideas.length > 0 && (
        <div className="space-y-4">
          {sections.ideas.map((idea, i) => (
            <IdeaCard key={i} idea={idea} index={i + 1} />
          ))}
        </div>
      )}

      {/* Community Pulse / Meta */}
      {sections.meta && (
        <div className="border-t pt-5 space-y-4">
          <h3 className="text-base font-semibold text-gray-800">
            Community Pulse
          </h3>
          {sections.meta.hotTopics.length > 0 && (
            <MetaSection title="Hot Topics" items={sections.meta.hotTopics} color="red" />
          )}
          {sections.meta.painPoints.length > 0 && (
            <MetaSection title="Pain Points" items={sections.meta.painPoints} color="amber" />
          )}
          {sections.meta.emergingTools.length > 0 && (
            <MetaSection title="Emerging Tools" items={sections.meta.emergingTools} color="blue" />
          )}
          {sections.meta.contrarianTakes.length > 0 && (
            <MetaSection title="Contrarian Takes" items={sections.meta.contrarianTakes} color="purple" />
          )}
        </div>
      )}

      {/* Fallback: raw markdown */}
      {sections.ideas.length === 0 && !sections.meta && content && (
        <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-wrap">
          {content}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface CommunityIdea {
  title: string;
  feasibility: string;
  excitement: number;
  problem: string;
  direction: string;
  whyCommunity: string;
  sourcePosts: string[];
}

interface CommunityMeta {
  hotTopics: string[];
  painPoints: string[];
  emergingTools: string[];
  contrarianTakes: string[];
}

function IdeaCard({ idea, index }: { idea: CommunityIdea; index: number }) {
  const feasEmoji: Record<string, string> = {
    QUICK_WIN: "\u26A1",
    PROJECT: "\uD83D\uDD2C",
    AMBITIOUS: "\uD83D\uDE80",
  };
  const stars =
    idea.excitement > 0
      ? "\u2605".repeat(idea.excitement) +
        "\u2606".repeat(5 - idea.excitement)
      : "";

  return (
    <div className="border rounded-xl p-4 hover:border-emerald-200 transition-colors">
      <div className="flex items-start justify-between gap-2 mb-2">
        <h4 className="font-semibold text-gray-900">
          {index}. {idea.title}
        </h4>
        <div className="flex items-center gap-2 flex-shrink-0 text-xs">
          {idea.feasibility && (
            <span className="px-2 py-0.5 bg-gray-100 rounded-full text-gray-600">
              {feasEmoji[idea.feasibility] || ""} {idea.feasibility}
            </span>
          )}
          {stars && (
            <span className="text-amber-500" title={`Excitement: ${idea.excitement}/5`}>
              {stars}
            </span>
          )}
        </div>
      </div>
      {idea.problem && (
        <p className="text-sm text-gray-600 mb-2">
          <span className="font-medium text-gray-700">Problem: </span>
          {idea.problem}
        </p>
      )}
      {idea.direction && (
        <p className="text-sm text-gray-600 mb-2">
          <span className="font-medium text-gray-700">Direction: </span>
          {idea.direction}
        </p>
      )}
      {idea.whyCommunity && (
        <p className="text-sm text-emerald-700 bg-emerald-50 rounded-lg px-3 py-2 mb-2">
          {idea.whyCommunity}
        </p>
      )}
      {idea.sourcePosts.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {idea.sourcePosts.map((s, i) => (
            <span
              key={i}
              className="text-[11px] text-gray-400 bg-gray-50 px-1.5 py-0.5 rounded"
            >
              {s.length > 60 ? s.slice(0, 57) + "..." : s}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function MetaSection({
  title,
  items,
  color,
}: {
  title: string;
  items: string[];
  color: string;
}) {
  const colorMap: Record<string, string> = {
    red: "bg-red-50 border-red-200",
    amber: "bg-amber-50 border-amber-200",
    blue: "bg-blue-50 border-blue-200",
    purple: "bg-purple-50 border-purple-200",
  };
  const dotMap: Record<string, string> = {
    red: "bg-red-400",
    amber: "bg-amber-400",
    blue: "bg-blue-400",
    purple: "bg-purple-400",
  };

  return (
    <div className={`rounded-lg border p-3 ${colorMap[color] || "bg-gray-50 border-gray-200"}`}>
      <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wider mb-2">
        {title}
      </h4>
      <ul className="space-y-1">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
            <span className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${dotMap[color] || "bg-gray-400"}`} />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Markdown parsing
// ---------------------------------------------------------------------------

function parseCommunitySections(content: string): {
  ideas: CommunityIdea[];
  meta: CommunityMeta | null;
} {
  const ideas: CommunityIdea[] = [];
  let meta: CommunityMeta | null = null;

  // Try to find idea sections: ## 1. Title
  const ideaRegex = /##\s+\d+\.\s+(.+?)(?=\n##\s+\d+\.|\n---|\n##\s+Community Pulse|$)/g;
  let match;

  while ((match = ideaRegex.exec(content)) !== null) {
    const block = match[0];
    const title = match[1].trim();

    const feasMatch = block.match(/\*\*Feasibility\*\*:\s*[^\s]*\s*(\w+)/);
    const exciteMatch = block.match(/\*\*Excitement\*\*:.*?\((\d)\/5\)/);
    const problemMatch = block.match(/\*\*Problem\*\*:\s*([\s\S]+?)(?=\n\n|\n\*\*)/);
    const directionMatch = block.match(/\*\*Research Direction\*\*:\s*([\s\S]+?)(?=\n\n|\n\*\*)/);
    const whyMatch = block.match(/\*\*Community Insight\*\*:\s*([\s\S]+?)(?=\n\n|\n\*\*)/);
    const sourcesMatch = block.match(/\*\*Inspired by\*\*:\s*([\s\S]+?)(?=\n\n|$)/);

    ideas.push({
      title,
      feasibility: feasMatch?.[1] || "",
      excitement: exciteMatch ? parseInt(exciteMatch[1]) : 0,
      problem: problemMatch?.[1]?.trim() || "",
      direction: directionMatch?.[1]?.trim() || "",
      whyCommunity: whyMatch?.[1]?.trim() || "",
      sourcePosts: sourcesMatch
        ? sourcesMatch[1]
            .split(" \u00B7 ")
            .map((s) => s.trim())
            .filter(Boolean)
        : [],
    });
  }

  // Parse meta section
  const metaStart = content.indexOf("## Community Pulse");
  if (metaStart >= 0) {
    const metaContent = content.slice(metaStart);
    meta = {
      hotTopics: extractListItems(metaContent, "### Hot Topics"),
      painPoints: extractListItems(metaContent, "### Pain Points"),
      emergingTools: extractListItems(metaContent, "### Emerging Tools"),
      contrarianTakes: extractListItems(metaContent, "### Contrarian Takes"),
    };
  }

  return { ideas, meta };
}

function extractListItems(text: string, heading: string): string[] {
  const start = text.indexOf(heading);
  if (start < 0) return [];
  const after = text.slice(start + heading.length);
  const end = after.search(/\n###|\n##/);
  const block = end >= 0 ? after.slice(0, end) : after;
  return block
    .split("\n")
    .map((l) => l.replace(/^[-*]\s*/, "").trim())
    .filter((l) => l.length > 0);
}
