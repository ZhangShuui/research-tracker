"use client";

import { ExternalLink, Trash2 } from "lucide-react";
import { Paper } from "@/lib/api";

interface Props {
  paper: Paper;
  onClick: () => void;
  onDelete?: (arxivId: string) => void;
}

export function PaperCard({ paper, onClick, onDelete }: Props) {
  return (
    <div
      onClick={onClick}
      className="bg-white rounded-xl border border-gray-200 p-4 hover:shadow-md transition-shadow cursor-pointer flex flex-col gap-2"
    >
      <div className="flex items-start justify-between gap-2">
        <h3 className="font-medium text-sm text-gray-900 leading-snug line-clamp-2">
          {paper.title}
        </h3>
        <div className="flex items-center gap-0.5 flex-shrink-0">
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="p-1 text-gray-400 hover:text-blue-600 transition-colors"
          >
            <ExternalLink size={14} />
          </a>
          {onDelete && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                if (confirm(`Delete "${paper.title.slice(0, 60)}…"?`)) {
                  onDelete(paper.arxiv_id);
                }
              }}
              className="p-1 text-gray-400 hover:text-red-600 transition-colors"
              title="Delete paper"
            >
              <Trash2 size={14} />
            </button>
          )}
        </div>
      </div>

      {paper.key_insight && (
        <p className="text-xs text-blue-700 bg-blue-50 rounded-lg px-2.5 py-1.5 leading-relaxed">
          {paper.key_insight}
        </p>
      )}

      {paper.method && (
        <p className="text-xs text-gray-500 line-clamp-2">
          <span className="font-medium text-gray-600">Method:</span>{" "}
          {paper.method}
        </p>
      )}

      <div className="flex flex-wrap gap-1.5 mt-auto pt-1">
        {paper.quality_score > 0 && (
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              paper.quality_score >= 4
                ? "bg-emerald-50 text-emerald-700"
                : paper.quality_score >= 3
                ? "bg-amber-50 text-amber-700"
                : "bg-red-50 text-red-700"
            }`}
          >
            Q{paper.quality_score}
          </span>
        )}
        {paper.venue && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-green-50 text-green-700 font-medium">
            {paper.venue}
          </span>
        )}
        {paper.math_concepts?.slice(0, 2).map((mc) => (
          <span
            key={mc}
            className="text-xs px-2 py-0.5 rounded-full bg-purple-50 text-purple-700"
          >
            {mc}
          </span>
        ))}
      </div>

      <p className="text-xs text-gray-400 mt-1">
        {paper.authors.split(",").slice(0, 3).join(",")}
        {paper.authors.split(",").length > 3 ? " et al." : ""}{" "}
        &middot; {new Date(paper.published).toLocaleDateString()}
      </p>
    </div>
  );
}
