"use client";

import { ResearchPlan } from "@/lib/api";
import { FileText, Clock, CheckCircle2, XCircle } from "lucide-react";

interface Props {
  plan: ResearchPlan;
  onClick: () => void;
  isSelected: boolean;
}

function extractAvgScore(review: string): number | null {
  const match = review.match(/Overall[:\s]*(\d+(?:\.\d+)?)\s*\/\s*10/gi);
  if (!match || match.length === 0) return null;
  let sum = 0;
  let count = 0;
  for (const m of match) {
    const num = m.match(/(\d+(?:\.\d+)?)/);
    if (num) {
      sum += parseFloat(num[1]);
      count++;
    }
  }
  return count > 0 ? sum / count : null;
}

function getScoreColor(score: number): {
  border: string;
  bg: string;
  text: string;
  ring: string;
} {
  if (score >= 6)
    return {
      border: "border-l-emerald-400",
      bg: "bg-emerald-50",
      text: "text-emerald-700",
      ring: "ring-emerald-200",
    };
  if (score >= 4)
    return {
      border: "border-l-amber-400",
      bg: "bg-amber-50",
      text: "text-amber-700",
      ring: "ring-amber-200",
    };
  return {
    border: "border-l-red-400",
    bg: "bg-red-50",
    text: "text-red-700",
    ring: "ring-red-200",
  };
}

const STATUS_CONFIG = {
  running: {
    icon: <Clock size={12} className="text-blue-500" />,
    dot: "bg-blue-500",
    label: "Running",
    labelColor: "text-blue-700 bg-blue-50 border-blue-200/60",
  },
  completed: {
    icon: <CheckCircle2 size={12} className="text-emerald-500" />,
    dot: "bg-emerald-500",
    label: "Done",
    labelColor: "text-emerald-700 bg-emerald-50 border-emerald-200/60",
  },
  failed: {
    icon: <XCircle size={12} className="text-red-400" />,
    dot: "bg-red-400",
    label: "Failed",
    labelColor: "text-red-600 bg-red-50 border-red-200/60",
  },
} as const;

const TOTAL_SECTIONS = 7;

export function ResearchPlanCard({ plan, onClick, isSelected }: Props) {
  const statusCfg = STATUS_CONFIG[plan.status];
  const avgScore = plan.review ? extractAvgScore(plan.review) : null;
  const scoreColors = avgScore != null ? getScoreColor(avgScore) : null;

  const sectionsDone = [
    plan.introduction,
    plan.related_work,
    plan.methodology,
    plan.experimental_design,
    plan.expected_results,
    plan.timeline,
    plan.review,
  ].filter(Boolean).length;

  const progressPct = (sectionsDone / TOTAL_SECTIONS) * 100;
  const isRunning = plan.status === "running";

  // Determine left border color based on score
  const leftBorderClass = scoreColors
    ? scoreColors.border
    : plan.status === "running"
    ? "border-l-blue-400"
    : plan.status === "failed"
    ? "border-l-red-300"
    : "border-l-slate-200";

  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-lg border border-l-[3px] transition-all duration-200 ${leftBorderClass} ${
        isSelected
          ? "border-indigo-200 bg-indigo-50/50 shadow-sm ring-1 ring-indigo-200/60"
          : "border-slate-200 bg-white hover:shadow-sm hover:border-slate-300"
      }`}
    >
      <div className="p-3">
        {/* Title row with score badge */}
        <div className="flex items-start justify-between gap-2 mb-1.5">
          <p className="text-[13px] font-semibold text-slate-800 line-clamp-2 leading-snug flex-1">
            {plan.idea_title || "Untitled Plan"}
          </p>
          {avgScore != null && scoreColors && (
            <span
              className={`flex-shrink-0 px-1.5 py-0.5 rounded text-[11px] font-bold tabular-nums ${scoreColors.bg} ${scoreColors.text} border ${scoreColors.ring}`}
            >
              {avgScore.toFixed(1)}
            </span>
          )}
        </div>

        {/* Status + meta compact row */}
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border ${statusCfg.labelColor}`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${statusCfg.dot} ${
                isRunning ? "animate-pulse" : ""
              }`}
            />
            {statusCfg.label}
          </span>
          <span className="text-[10px] text-slate-400 flex items-center gap-1">
            <FileText size={9} />
            {sectionsDone}/{TOTAL_SECTIONS}
          </span>
          {plan.started_at && (
            <span className="text-[10px] text-slate-400">
              {new Date(plan.started_at).toLocaleDateString(undefined, {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          )}
        </div>

        {/* Progress bar (only when running) */}
        {isRunning && (
          <div className="mt-2">
            <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-blue-400 animate-[progressPulse_2s_ease-in-out_infinite] transition-all duration-700"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </button>
  );
}
