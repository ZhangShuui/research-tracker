"use client";

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface ReviewerScore {
  name: string;
  soundness: number;
  presentation: number;
  contribution: number;
  overall: number;
  confidence: number;
}

function parseReviewScores(review: string): ReviewerScore[] {
  const reviewers: ReviewerScore[] = [];
  const reviewerBlocks = review.split(/###\s+Reviewer\s+\d+/i).slice(1);

  for (let i = 0; i < reviewerBlocks.length && i < 3; i++) {
    const block = reviewerBlocks[i];
    const name =
      block.match(/^[:\s]*(.+?)(?:\n|$)/)?.[1]?.trim() ||
      `Reviewer ${i + 1}`;

    const soundness = parseFloat(
      block.match(/Soundness[:\s]*\[?(\d+(?:\.\d+)?)/i)?.[1] || "0"
    );
    const presentation = parseFloat(
      block.match(/Presentation[:\s]*\[?(\d+(?:\.\d+)?)/i)?.[1] || "0"
    );
    const contribution = parseFloat(
      block.match(/Contribution[:\s]*\[?(\d+(?:\.\d+)?)/i)?.[1] || "0"
    );
    const overall = parseFloat(
      block.match(/Overall[:\s]*\[?(\d+(?:\.\d+)?)/i)?.[1] || "0"
    );
    const confidence = parseFloat(
      block.match(/Confidence[:\s]*\[?(\d+(?:\.\d+)?)/i)?.[1] || "0"
    );

    if (soundness || presentation || contribution || overall) {
      reviewers.push({
        name,
        soundness,
        presentation,
        contribution,
        overall,
        confidence,
      });
    }
  }

  return reviewers;
}

const REVIEWER_COLORS = ["#3b82f6", "#f59e0b", "#10b981"];

function getRecommendation(avgOverall: number): {
  label: string;
  color: string;
  bg: string;
  border: string;
} {
  if (avgOverall >= 6.5) {
    return {
      label: "Accept",
      color: "text-emerald-700",
      bg: "bg-emerald-50",
      border: "border-emerald-200",
    };
  }
  if (avgOverall >= 4.5) {
    return {
      label: "Borderline",
      color: "text-amber-700",
      bg: "bg-amber-50",
      border: "border-amber-200",
    };
  }
  return {
    label: "Reject",
    color: "text-red-700",
    bg: "bg-red-50",
    border: "border-red-200",
  };
}

function ScoreBar({
  label,
  value,
  max,
}: {
  label: string;
  value: number;
  max: number;
}) {
  const pct = Math.min((value / max) * 100, 100);
  const color =
    pct >= 70 ? "bg-emerald-500" : pct >= 50 ? "bg-amber-500" : "bg-red-400";

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 text-slate-500 text-right">{label}</span>
      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-10 text-slate-600 font-medium tabular-nums">
        {value}/{max}
      </span>
    </div>
  );
}

export function ReviewScorePanel({ review }: { review: string }) {
  const reviewers = parseReviewScores(review);

  if (reviewers.length === 0) return null;

  const avgOverall =
    reviewers.reduce((s, r) => s + r.overall, 0) / reviewers.length;
  const recommendation = getRecommendation(avgOverall);

  // Build radar chart data: scale Overall from /10 to /4 for visual parity
  const radarData = [
    {
      dimension: "Soundness",
      ...Object.fromEntries(
        reviewers.map((r, i) => [`r${i}`, r.soundness])
      ),
    },
    {
      dimension: "Presentation",
      ...Object.fromEntries(
        reviewers.map((r, i) => [`r${i}`, r.presentation])
      ),
    },
    {
      dimension: "Contribution",
      ...Object.fromEntries(
        reviewers.map((r, i) => [`r${i}`, r.contribution])
      ),
    },
    {
      dimension: "Overall",
      ...Object.fromEntries(
        reviewers.map((r, i) => [`r${i}`, Math.round((r.overall / 10) * 4 * 10) / 10])
      ),
    },
  ];

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-5">
      {/* Header with average score and recommendation */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-700 tracking-tight">
            Peer Review Summary
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            {reviewers.length} reviewer{reviewers.length !== 1 ? "s" : ""}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 text-xs font-semibold rounded-full border ${recommendation.color} ${recommendation.bg} ${recommendation.border}`}
          >
            {recommendation.label}
          </span>
          <span
            className={`text-lg font-bold tabular-nums ${
              avgOverall >= 6
                ? "text-emerald-600"
                : avgOverall >= 4
                ? "text-amber-600"
                : "text-red-500"
            }`}
          >
            {avgOverall.toFixed(1)}/10
          </span>
        </div>
      </div>

      {/* Radar chart */}
      <div className="flex justify-center">
        <div className="w-full max-w-sm h-56">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="72%">
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis
                dataKey="dimension"
                tick={{ fontSize: 11, fill: "#64748b" }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 4]}
                tickCount={5}
                tick={{ fontSize: 9, fill: "#94a3b8" }}
              />
              {reviewers.map((r, i) => (
                <Radar
                  key={i}
                  name={r.name}
                  dataKey={`r${i}`}
                  stroke={REVIEWER_COLORS[i]}
                  fill={REVIEWER_COLORS[i]}
                  fillOpacity={0.1}
                  strokeWidth={1.5}
                />
              ))}
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                iconSize={8}
                iconType="circle"
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Individual reviewer details */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {reviewers.map((r, i) => (
          <div
            key={i}
            className="space-y-2 p-3 bg-slate-50/80 rounded-lg border border-slate-100"
          >
            <div className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: REVIEWER_COLORS[i] }}
              />
              <p className="text-xs font-semibold text-slate-700 truncate">
                {r.name}
              </p>
            </div>
            <ScoreBar label="Soundness" value={r.soundness} max={4} />
            <ScoreBar label="Presentation" value={r.presentation} max={4} />
            <ScoreBar label="Contribution" value={r.contribution} max={4} />
            <ScoreBar label="Overall" value={r.overall} max={10} />
            <div className="text-[10px] text-slate-400 text-right">
              Confidence: {r.confidence}/5
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
