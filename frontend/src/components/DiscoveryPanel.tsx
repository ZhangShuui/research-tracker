"use client";

import { MathMarkdown } from "./MathMarkdown";
import {
  Clock,
  FileText,
  BarChart3,
  Shield,
  AlertTriangle,
  CheckCircle,
  ExternalLink,
  Sparkles,
  TrendingUp,
  Zap,
  BarChart,
  Activity,
  FlaskConical,
} from "lucide-react";
import type { DiscoveryReport } from "@/lib/api";

export interface ThemeInfo {
  title: string;
  description: string;
  techniques: string[];
  papers: string[];
  trend?: string;
}

interface Props {
  report: DiscoveryReport;
  onReview?: () => void;
  onRegenerate?: () => void;
  onResearchTheme?: (theme: ThemeInfo) => void;
  isReviewing?: boolean;
  isRegenerating?: boolean;
}

/* ── Quality badge ─────────────────────────────────────────── */

function QualityBadge({ score }: { score: number }) {
  if (score < 0) return null;
  const color =
    score >= 80
      ? "bg-emerald-100 text-emerald-700 border-emerald-200"
      : score >= 60
      ? "bg-amber-100 text-amber-700 border-amber-200"
      : "bg-red-100 text-red-700 border-red-200";
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${color}`}
    >
      {score >= 80 ? <CheckCircle size={12} /> : <AlertTriangle size={12} />}
      Quality: {score}/100
    </span>
  );
}

/* ── Trend direction badge ─────────────────────────────────── */

const TREND_META: Record<
  string,
  { icon: React.ReactNode; color: string; label: string }
> = {
  EMERGING: {
    icon: <Sparkles size={12} />,
    color: "bg-teal-50 text-teal-700 border-teal-200",
    label: "Emerging",
  },
  ACCELERATING: {
    icon: <TrendingUp size={12} />,
    color: "bg-blue-50 text-blue-700 border-blue-200",
    label: "Accelerating",
  },
  ESTABLISHED: {
    icon: <BarChart size={12} />,
    color: "bg-gray-100 text-gray-600 border-gray-200",
    label: "Established",
  },
  PEAKING: {
    icon: <Activity size={12} />,
    color: "bg-orange-50 text-orange-700 border-orange-200",
    label: "Peaking",
  },
};

/* ── Parse markdown into sections ──────────────────────────── */

interface ThemeSection {
  title: string;
  trend?: string;
  body: string;
  techniques: string[];
  papers: string[];
}

interface CrossTheme {
  title: string;
  content: string;
}

function parseTrendingSections(md: string) {
  // Strip the header block (# heading + meta + ---)
  const headerEnd = md.indexOf("---\n");
  const body = headerEnd >= 0 ? md.slice(headerEnd + 4).trim() : md;

  const themes: ThemeSection[] = [];
  const crossThemes: CrossTheme[] = [];
  let inCross = false;

  // Split by ## headings
  const parts = body.split(/^## /m).filter(Boolean);

  for (const part of parts) {
    const lines = part.trim().split("\n");
    const heading = lines[0]?.replace(/^\d+\.\s*/, "").trim() ?? "";
    const rest = lines.slice(1).join("\n").trim();

    if (/cross.theme/i.test(heading)) {
      inCross = true;
      // parse sub-sections (### headings)
      const subs = rest.split(/^### /m).filter(Boolean);
      for (const sub of subs) {
        const subLines = sub.trim().split("\n");
        crossThemes.push({
          title: subLines[0]?.trim() ?? "",
          content: subLines.slice(1).join("\n").trim(),
        });
      }
      continue;
    }

    if (inCross) {
      // Sometimes cross-theme sub-sections are at ## level
      crossThemes.push({ title: heading, content: rest });
      continue;
    }

    // Parse theme section
    let trend = "";
    const trendMatch = rest.match(/\*\*Trend\*\*:\s*[^\s]*\s*(\w+)/);
    if (trendMatch) trend = trendMatch[1];

    const techMatch = rest.match(/\*\*Key Techniques\*\*:\s*(.+)/);
    const techniques = techMatch
      ? techMatch[1]
          .split("·")
          .map((t) => t.replace(/`/g, "").trim())
          .filter(Boolean)
      : [];

    const paperMatch = rest.match(/\*\*Representative Papers\*\*:\s*(.+)/);
    const papers: string[] = [];
    if (paperMatch) {
      let m: RegExpExecArray | null;
      const re = /\[([^\]]+)\]/g;
      while ((m = re.exec(paperMatch[1])) !== null) {
        papers.push(m[1]);
      }
    }

    // Clean body: remove the parsed lines
    const bodyText = rest
      .replace(/\*\*Trend\*\*:.+\n?/g, "")
      .replace(/\*\*Key Techniques\*\*:.+\n?/g, "")
      .replace(/\*\*Representative Papers\*\*:.+\n?/g, "")
      .trim();

    themes.push({ title: heading, trend, body: bodyText, techniques, papers });
  }

  return { themes, crossThemes };
}

/* ── Parse math sections ───────────────────────────────────── */

interface MathPaper {
  id: string;
  title: string;
  elegance: number;
  coreConcepts: string;
  techniques: string[];
  mlApps: string;
}

interface MathSynthesis {
  title: string;
  items: string[];
  text: string;
}

function parseMathSections(md: string) {
  const headerEnd = md.indexOf("---\n");
  const body = headerEnd >= 0 ? md.slice(headerEnd + 4).trim() : md;

  const papers: MathPaper[] = [];
  const synthesisSections: MathSynthesis[] = [];
  let inSynthesis = false;

  const parts = body.split(/^## /m).filter(Boolean);

  for (const part of parts) {
    const lines = part.trim().split("\n");
    const heading = lines[0]?.trim() ?? "";
    const rest = lines.slice(1).join("\n").trim();

    if (/synthesis/i.test(heading)) {
      inSynthesis = true;
      const subs = rest.split(/^### /m).filter(Boolean);
      for (const sub of subs) {
        const subLines = sub.trim().split("\n");
        const title = subLines[0]?.trim() ?? "";
        const content = subLines.slice(1).join("\n").trim();
        const items: string[] = [];
        const numbered: string[] = [];
        {
          let m: RegExpExecArray | null;
          const re1 = /^[-•]\s*(.+)$/gm;
          while ((m = re1.exec(content)) !== null) items.push(m[1].trim());
          const re2 = /^\d+\.\s*(.+)$/gm;
          while ((m = re2.exec(content)) !== null) numbered.push(m[1].trim());
        }
        synthesisSections.push({
          title,
          items: items.length > 0 ? items : numbered,
          text: items.length === 0 && numbered.length === 0 ? content : "",
        });
      }
      continue;
    }

    if (inSynthesis) {
      synthesisSections.push({ title: heading, items: [], text: rest });
      continue;
    }

    if (/paper anal/i.test(heading)) {
      // Parse ### sub-sections for each paper
      const paperParts = rest.split(/^### /m).filter(Boolean);
      for (const pp of paperParts) {
        const ppLines = pp.trim().split("\n");
        const ppHeading = ppLines[0]?.trim() ?? "";
        const ppRest = ppLines.slice(1).join("\n").trim();

        // Extract arxiv ID from heading like "1. [2603.02844](url) — Title"
        const idMatch = ppHeading.match(
          /\d+\.\s*\[([^\]]+)\](?:\([^)]*\))?\s*(?:—\s*(.+))?/
        );
        const id = idMatch?.[1] ?? ppHeading.replace(/^\d+\.\s*/, "");
        const title = idMatch?.[2] ?? "";

        // Elegance
        const eleganceMatch = ppRest.match(
          /\*\*Elegance\*\*:\s*[★☆]+\s*\((\d)\/5\)/
        );
        const elegance = eleganceMatch ? parseInt(eleganceMatch[1]) : 0;

        const coreMatch = ppRest.match(
          /\*\*Core Concepts?\*\*:\s*([\s\S]+?)(?:\n\n|\n\*\*|$)/
        );
        const coreConcepts = coreMatch?.[1]?.trim() ?? "";

        const techMatch = ppRest.match(/\*\*Techniques?\*\*:\s*(.+)/);
        const techniques = techMatch
          ? techMatch[1]
              .split("·")
              .map((t) => t.replace(/`/g, "").trim())
              .filter(Boolean)
          : [];

        const mlMatch = ppRest.match(
          /\*\*ML Applications?\*\*:\s*([\s\S]+?)(?:\n\n|\n\*\*|$)/
        );
        const mlApps = mlMatch?.[1]?.trim() ?? "";

        papers.push({ id, title, elegance, coreConcepts, techniques, mlApps });
      }
      continue;
    }
  }

  return { papers, synthesisSections };
}

/* ── Theme Card ────────────────────────────────────────────── */

function ThemeCard({
  theme,
  index,
  onResearch,
}: {
  theme: ThemeSection;
  index: number;
  onResearch?: (info: ThemeInfo) => void;
}) {
  const trendMeta = theme.trend ? TREND_META[theme.trend] : null;

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 space-y-3 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-base font-semibold text-gray-900 leading-snug">
          <span className="text-indigo-500 mr-1.5">{index}.</span>
          {theme.title}
        </h3>
        <div className="flex items-center gap-2 flex-shrink-0">
          {trendMeta && (
            <span
              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border whitespace-nowrap ${trendMeta.color}`}
            >
              {trendMeta.icon}
              {trendMeta.label}
            </span>
          )}
        </div>
      </div>

      {theme.body && (
        <p className="text-sm text-gray-600 leading-relaxed">{theme.body}</p>
      )}

      {theme.techniques.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {theme.techniques.map((t) => (
            <span
              key={t}
              className="px-2 py-0.5 text-xs rounded-md bg-indigo-50 text-indigo-700 font-medium"
            >
              {t}
            </span>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between pt-1">
        {theme.papers.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {theme.papers.map((p) => (
              <a
                key={p}
                href={`https://arxiv.org/abs/${p}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-indigo-600 transition-colors"
              >
                <ExternalLink size={10} />
                {p}
              </a>
            ))}
          </div>
        ) : (
          <div />
        )}

        {onResearch && (
          <button
            onClick={() =>
              onResearch({
                title: theme.title,
                description: theme.body,
                techniques: theme.techniques,
                papers: theme.papers,
                trend: theme.trend,
              })
            }
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors whitespace-nowrap"
          >
            <FlaskConical size={12} />
            Research This
          </button>
        )}
      </div>
    </div>
  );
}

/* ── Math Paper Card ───────────────────────────────────────── */

function MathPaperCard({ paper, index }: { paper: MathPaper; index: number }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 space-y-2.5 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-sm font-semibold text-gray-900 leading-snug">
          <span className="text-purple-500 mr-1.5">{index}.</span>
          <a
            href={`https://arxiv.org/abs/${paper.id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-purple-600 transition-colors"
          >
            {paper.id}
          </a>
          {paper.title && (
            <span className="font-normal text-gray-500 ml-1.5">
              — {paper.title}
            </span>
          )}
        </h3>
        {paper.elegance > 0 && (
          <span className="text-xs whitespace-nowrap text-amber-500 font-medium">
            {"★".repeat(paper.elegance)}
            {"☆".repeat(5 - paper.elegance)}
          </span>
        )}
      </div>

      {paper.coreConcepts && (
        <p className="text-sm text-gray-600">
          <span className="font-medium text-gray-700">Core: </span>
          {paper.coreConcepts}
        </p>
      )}

      {paper.techniques.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {paper.techniques.map((t) => (
            <span
              key={t}
              className="px-2 py-0.5 text-xs rounded-md bg-purple-50 text-purple-700 font-medium"
            >
              {t}
            </span>
          ))}
        </div>
      )}

      {paper.mlApps && (
        <p className="text-sm text-gray-600">
          <span className="font-medium text-gray-700">ML: </span>
          {paper.mlApps}
        </p>
      )}
    </div>
  );
}

/* ── Main Component ────────────────────────────────────────── */

export function DiscoveryPanel({
  report,
  onReview,
  onRegenerate,
  onResearchTheme,
  isReviewing,
  isRegenerating,
}: Props) {
  const isRunning = report.status === "running";
  const isFailed = report.status === "failed";
  const isCompleted = report.status === "completed";
  const hasQuality = report.quality_score >= 0;
  const isLowQuality = hasQuality && report.quality_score < 60;

  const isTrending = report.type === "trending";
  const isMath = report.type === "math";

  // Parse sections
  const trendingData = isTrending && isCompleted
    ? parseTrendingSections(report.content ?? "")
    : null;
  const mathData = isMath && isCompleted
    ? parseMathSections(report.content ?? "")
    : null;

  const hasStructured =
    (trendingData && trendingData.themes.length > 0) ||
    (mathData && mathData.papers.length > 0);

  return (
    <div className="space-y-5">
      {/* Status bar */}
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-medium ${
            isRunning
              ? "bg-amber-100 text-amber-700"
              : isFailed
              ? "bg-red-100 text-red-700"
              : "bg-emerald-100 text-emerald-700"
          }`}
        >
          {isRunning && (
            <span className="w-2 h-2 bg-amber-500 rounded-full animate-pulse" />
          )}
          {report.status}
        </span>

        <QualityBadge score={report.quality_score} />

        {report.started_at && (
          <span className="flex items-center gap-1 text-gray-500">
            <Clock size={14} />
            {new Date(report.started_at).toLocaleString()}
          </span>
        )}

        <span className="flex items-center gap-1 text-gray-500">
          <FileText size={14} />
          {report.paper_count} papers
        </span>

        {report.source_stats &&
          Object.keys(report.source_stats).length > 0 && (
            <span className="flex items-center gap-1 text-gray-500">
              <BarChart3 size={14} />
              {Object.entries(report.source_stats)
                .map(([k, v]) => `${k}: ${v}`)
                .join(", ")}
            </span>
          )}
      </div>

      {/* Quality flags */}
      {hasQuality &&
        report.quality_flags &&
        report.quality_flags.length > 0 && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 space-y-1.5">
            <h4 className="text-xs font-semibold text-amber-700 uppercase tracking-wider flex items-center gap-1">
              <AlertTriangle size={12} />
              Quality Issues
            </h4>
            {report.quality_flags.map((flag, i) => (
              <div key={i} className="flex items-start gap-2 text-sm">
                <span
                  className={`mt-0.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                    flag.severity === "high"
                      ? "bg-red-500"
                      : flag.severity === "medium"
                      ? "bg-amber-500"
                      : "bg-gray-400"
                  }`}
                />
                <span className="text-amber-800">{flag.issue}</span>
              </div>
            ))}
          </div>
        )}

      {/* Action buttons */}
      {isCompleted && (onReview || onRegenerate) && (
        <div className="flex items-center gap-2">
          {onReview && !hasQuality && (
            <button
              onClick={onReview}
              disabled={isReviewing}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50"
            >
              <Shield size={14} />
              {isReviewing ? "Reviewing..." : "Review Quality"}
            </button>
          )}
          {onRegenerate && isLowQuality && (
            <button
              onClick={onRegenerate}
              disabled={isRegenerating}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-amber-600 text-white hover:bg-amber-700 transition-colors disabled:opacity-50"
            >
              {isRegenerating ? "Re-generating..." : "Re-generate (Low Quality)"}
            </button>
          )}
        </div>
      )}

      {/* ── Content ─────────────────────────────────────────── */}
      {isRunning ? (
        <div className="flex items-center gap-3 py-12 justify-center text-gray-500">
          <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span>Analyzing papers... this may take a few minutes.</span>
        </div>
      ) : hasStructured ? (
        <>
          {/* ── Trending Cards ── */}
          {trendingData && trendingData.themes.length > 0 && (
            <div className="space-y-6">
              <div className="grid gap-4">
                {trendingData.themes.map((theme, i) => (
                  <ThemeCard
                    key={i}
                    theme={theme}
                    index={i + 1}
                    onResearch={onResearchTheme}
                  />
                ))}
              </div>

              {trendingData.crossThemes.length > 0 && (
                <div className="rounded-xl border border-indigo-100 bg-indigo-50/50 p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-indigo-800 uppercase tracking-wider flex items-center gap-1.5">
                    <Zap size={14} />
                    Cross-Theme Observations
                  </h3>
                  {trendingData.crossThemes.map((ct, i) => (
                    <div key={i}>
                      <h4 className="text-sm font-medium text-indigo-700 mb-1">
                        {ct.title}
                      </h4>
                      <MathMarkdown className="text-sm text-gray-700 leading-relaxed prose prose-sm max-w-none">{ct.content}</MathMarkdown>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── Math Paper Cards ── */}
          {mathData && mathData.papers.length > 0 && (
            <div className="space-y-6">
              <div className="grid gap-3">
                {mathData.papers.map((paper, i) => (
                  <MathPaperCard key={i} paper={paper} index={i + 1} />
                ))}
              </div>

              {mathData.synthesisSections.length > 0 && (
                <div className="rounded-xl border border-purple-100 bg-purple-50/50 p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-purple-800 uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkles size={14} />
                    Synthesis
                  </h3>
                  {mathData.synthesisSections.map((ss, i) => (
                    <div key={i}>
                      <h4 className="text-sm font-medium text-purple-700 mb-1">
                        {ss.title}
                      </h4>
                      {ss.items.length > 0 ? (
                        <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                          {ss.items.map((item, j) => (
                            <li key={j}>
                              <MathMarkdown>{item}</MathMarkdown>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <MathMarkdown className="text-sm text-gray-700 leading-relaxed prose prose-sm max-w-none">{ss.text}</MathMarkdown>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      ) : report.content ? (
        /* Fallback: render as markdown */
        <MathMarkdown className="prose prose-sm max-w-none text-gray-700 prose-headings:text-gray-900 prose-strong:text-gray-900 prose-code:text-indigo-600 prose-code:bg-indigo-50 prose-code:px-1 prose-code:py-0.5 prose-code:rounded">{report.content}</MathMarkdown>
      ) : (
        <p className="text-sm text-gray-400 italic py-8 text-center">
          No content available.
        </p>
      )}
    </div>
  );
}
