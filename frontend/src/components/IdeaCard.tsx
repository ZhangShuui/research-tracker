"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  ChevronDown,
  ChevronUp,
  FileText,
  Search,
  Loader2,
  ExternalLink,
} from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, BrainstormIdea, PriorArtResult } from "@/lib/api";
import { TranslateButton, TranslatedBlock } from "./TranslateButton";

interface Props {
  idea: BrainstormIdea;
  index: number;
  brainstormSessionId?: string;
}

function ScoreBadge({ label, score }: { label: string; score: number }) {
  const color =
    score >= 7
      ? "bg-green-50 text-green-700"
      : score >= 4
      ? "bg-yellow-50 text-yellow-700"
      : "bg-red-50 text-red-700";
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${color}`}>
      {label}: {score}/10
    </span>
  );
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const map: Record<string, string> = {
    NOVEL: "bg-green-100 text-green-800",
    PARTIALLY_NOVEL: "bg-yellow-100 text-yellow-800",
    ALREADY_EXISTS: "bg-red-100 text-red-800",
  };
  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full font-medium ${
        map[verdict] ?? "bg-gray-100 text-gray-600"
      }`}
    >
      {verdict?.replace("_", " ") || "PENDING"}
    </span>
  );
}

function MaturityBadge({ level }: { level: string }) {
  const styles: Record<string, string> = {
    NASCENT: "bg-blue-100 text-blue-800",
    GROWING: "bg-teal-100 text-teal-800",
    MATURE: "bg-amber-100 text-amber-800",
    SATURATED: "bg-red-100 text-red-800",
  };
  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full font-medium ${
        styles[level] ?? "bg-gray-100 text-gray-600"
      }`}
    >
      {level}
    </span>
  );
}

function RecommendationBadge({ rec }: { rec: string }) {
  const styles: Record<string, string> = {
    PURSUE: "bg-green-100 text-green-800",
    DIFFERENTIATE: "bg-yellow-100 text-yellow-800",
    RECONSIDER: "bg-red-100 text-red-800",
  };
  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full font-medium ${
        styles[rec] ?? "bg-gray-100 text-gray-600"
      }`}
    >
      {rec}
    </span>
  );
}

function PriorArtDisplay({ data }: { data: PriorArtResult }) {
  return (
    <div className="space-y-3 mt-3 pt-3 border-t border-gray-100">
      <div>
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
          Novelty Assessment
        </h4>
        <p className="text-sm text-gray-700 leading-relaxed">
          {data.novelty_assessment}
        </p>
      </div>

      <div>
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
          Recommendation
        </h4>
        <p className="text-sm text-gray-700 leading-relaxed">
          {data.recommendation_reason}
        </p>
      </div>

      {data.prior_works.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
            Prior Works ({data.prior_works.length})
          </h4>
          <ul className="space-y-1">
            {data.prior_works.map((w) => (
              <li key={w.arxiv_id} className="text-sm">
                <a
                  href={`https://arxiv.org/abs/${w.arxiv_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline inline-flex items-center gap-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  {w.title}
                  <ExternalLink size={10} />
                </a>
                {w.relevance && (
                  <span className="text-gray-500 text-xs ml-1">
                    — {w.relevance}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {data.similar_works.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
            Similar Works ({data.similar_works.length})
          </h4>
          <ul className="space-y-1">
            {data.similar_works.map((w) => (
              <li key={w.arxiv_id} className="text-sm">
                <a
                  href={`https://arxiv.org/abs/${w.arxiv_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline inline-flex items-center gap-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  {w.title}
                  <ExternalLink size={10} />
                </a>
                {w.overlap && (
                  <span className="text-gray-500 text-xs ml-1">
                    — {w.overlap}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function IdeaCard({ idea, index, brainstormSessionId }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [translatedText, setTranslatedText] = useState<string | null>(null);
  const { id: topicId } = useParams<{ id: string }>();
  const router = useRouter();
  const qc = useQueryClient();

  const priorArtMut = useMutation({
    mutationFn: () =>
      api.checkPriorArt(topicId, brainstormSessionId!, index),
    onSuccess: () => {
      qc.invalidateQueries({
        queryKey: ["brainstorm-detail", topicId, brainstormSessionId],
      });
    },
  });

  function handleGeneratePlan(e: React.MouseEvent) {
    e.stopPropagation();
    // Store idea in sessionStorage to avoid URL length limits (431 error)
    sessionStorage.setItem("prefill-idea", JSON.stringify(idea));
    const bsParam = brainstormSessionId ? `&bs_id=${brainstormSessionId}` : "";
    router.push(`/topics/${topicId}/research-plan?from_idea=1${bsParam}`);
  }

  function handleCheckPriorArt(e: React.MouseEvent) {
    e.stopPropagation();
    priorArtMut.mutate();
  }

  const pa = idea.prior_art;
  const isDropped = idea.status === "dropped";

  return (
    <div className={`rounded-xl border overflow-hidden ${
      isDropped
        ? "bg-gray-50 border-gray-200 opacity-70"
        : "bg-white border-gray-200"
    }`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start gap-3 p-4 text-left hover:bg-gray-50 transition-colors"
      >
        <span className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
          isDropped
            ? "bg-red-100 text-red-600"
            : "bg-blue-100 text-blue-700"
        }`}>
          {index + 1}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className={`font-medium text-sm ${isDropped ? "text-gray-500" : "text-gray-900"}`}>
              {idea.title || "Untitled Idea"}
            </h3>
            {isDropped && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-100 text-red-700 font-semibold uppercase">
                Dropped
              </span>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">
            {idea.problem}
          </p>
          <div className="flex flex-wrap gap-2 mt-2">
            {idea.novelty_score != null && (
              <ScoreBadge label="Novelty" score={idea.novelty_score} />
            )}
            {idea.feasibility_score != null && (
              <ScoreBadge label="Feasibility" score={idea.feasibility_score} />
            )}
            {idea.novelty_verdict && (
              <VerdictBadge verdict={idea.novelty_verdict} />
            )}
            {pa && <MaturityBadge level={pa.maturity_level} />}
            {pa && <RecommendationBadge rec={pa.recommendation} />}
          </div>
        </div>
        <span className="flex-shrink-0 mt-1 text-gray-400">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </span>
      </button>

      {expanded && (
        <div className="border-t border-gray-100 p-4 space-y-3 text-sm">
          {/* Review feedback for dropped ideas */}
          {isDropped && idea.review && (
            <ReviewFeedback review={idea.review} />
          )}

          {idea.motivation && (
            <Field title="Motivation">{idea.motivation}</Field>
          )}
          {idea.method && <Field title="Method">{idea.method}</Field>}
          {idea.experiment_plan && (
            <Field title="Experiment Plan">{idea.experiment_plan}</Field>
          )}

          <div className="flex flex-wrap gap-2 mt-2">
            {!pa && brainstormSessionId && (
              <button
                onClick={handleCheckPriorArt}
                disabled={priorArtMut.isPending}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-gradient-to-r from-violet-600 to-indigo-600 text-white rounded-lg hover:from-violet-700 hover:to-indigo-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {priorArtMut.isPending ? (
                  <Loader2 size={12} className="animate-spin" />
                ) : (
                  <Search size={12} />
                )}
                {priorArtMut.isPending
                  ? "Checking Prior Art..."
                  : "Check Prior Art"}
              </button>
            )}
            <button
              onClick={handleGeneratePlan}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-gradient-to-r from-emerald-600 to-blue-600 text-white rounded-lg hover:from-emerald-700 hover:to-blue-700 transition-all"
            >
              <FileText size={12} />
              Generate Research Plan
            </button>
            <TranslateButton
              sourceType="brainstorm_idea"
              sourceId={`${brainstormSessionId || "unknown"}_${index}`}
              field="full"
              content={[idea.motivation, idea.method, idea.experiment_plan]
                .filter(Boolean)
                .join("\n\n")}
              onTranslated={setTranslatedText}
            />
          </div>

          {translatedText && <TranslatedBlock content={translatedText} />}

          {priorArtMut.isError && (
            <p className="text-xs text-red-600 mt-1">
              Failed to check prior art:{" "}
              {(priorArtMut.error as Error).message}
            </p>
          )}

          {pa && <PriorArtDisplay data={pa} />}
        </div>
      )}
    </div>
  );
}

function ReviewFeedback({ review }: { review: NonNullable<BrainstormIdea["review"]> }) {
  const scores = [
    { label: "Novelty", value: review.novelty },
    { label: "Feasibility", value: review.feasibility },
    { label: "Clarity", value: review.clarity },
    { label: "Impact", value: review.impact },
  ].filter((s) => s.value != null);

  return (
    <div className="bg-red-50 border border-red-100 rounded-lg p-3 space-y-2">
      <h4 className="text-xs font-semibold text-red-700 uppercase tracking-wide">
        Drop Reason
      </h4>
      {scores.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {scores.map((s) => (
            <span
              key={s.label}
              className={`text-[11px] px-1.5 py-0.5 rounded font-medium ${
                (s.value ?? 0) <= 3
                  ? "bg-red-100 text-red-700"
                  : (s.value ?? 0) <= 5
                  ? "bg-yellow-100 text-yellow-700"
                  : "bg-green-100 text-green-700"
              }`}
            >
              {s.label}: {s.value}/10
            </span>
          ))}
          {review.overall != null && (
            <span className="text-[11px] px-1.5 py-0.5 rounded font-semibold bg-gray-100 text-gray-700">
              Overall: {review.overall}
            </span>
          )}
        </div>
      )}
      {review.weaknesses && review.weaknesses.length > 0 && (
        <ul className="text-xs text-red-800 space-y-0.5 list-disc list-inside">
          {review.weaknesses.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Field({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
        {title}
      </h4>
      <p className="text-gray-700 leading-relaxed">{children}</p>
    </div>
  );
}
