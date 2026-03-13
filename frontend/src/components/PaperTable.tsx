"use client";

import { ExternalLink, Trash2 } from "lucide-react";
import { Paper } from "@/lib/api";

interface Props {
  papers: Paper[];
  onSelect: (paper: Paper) => void;
  onDelete?: (arxivId: string) => void;
}

export function PaperTable({ papers, onSelect, onDelete }: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-gray-500 border-b">
            <th className="pb-2 pr-3 font-medium">Title</th>
            <th className="pb-2 pr-3 font-medium w-32">Key Insight</th>
            <th className="pb-2 pr-3 font-medium w-20">Venue</th>
            <th className="pb-2 pr-3 font-medium w-12">Score</th>
            <th className="pb-2 pr-3 font-medium w-24">Date</th>
            <th className="pb-2 font-medium w-16"></th>
          </tr>
        </thead>
        <tbody>
          {papers.map((p) => (
            <tr
              key={p.arxiv_id}
              onClick={() => onSelect(p)}
              className="border-b cursor-pointer hover:bg-gray-50 transition-colors"
            >
              <td className="py-2.5 pr-3">
                <span className="text-gray-900 line-clamp-1 font-medium">
                  {p.title}
                </span>
                <span className="text-xs text-gray-400 block mt-0.5">
                  {p.authors.split(",").slice(0, 2).join(",")}
                  {p.authors.split(",").length > 2 ? " et al." : ""}
                </span>
              </td>
              <td className="py-2.5 pr-3 text-xs text-gray-600 line-clamp-2">
                {p.key_insight || p.summary?.slice(0, 80)}
              </td>
              <td className="py-2.5 pr-3">
                {p.venue && (
                  <span className="text-xs px-1.5 py-0.5 rounded bg-green-50 text-green-700">
                    {p.venue}
                  </span>
                )}
              </td>
              <td className="py-2.5 pr-3">
                {p.quality_score > 0 && (
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                      p.quality_score >= 4
                        ? "bg-emerald-50 text-emerald-700"
                        : p.quality_score >= 3
                        ? "bg-amber-50 text-amber-700"
                        : "bg-red-50 text-red-700"
                    }`}
                  >
                    Q{p.quality_score}
                  </span>
                )}
              </td>
              <td className="py-2.5 pr-3 text-xs text-gray-500 whitespace-nowrap">
                {new Date(p.published).toLocaleDateString()}
              </td>
              <td className="py-2.5">
                <div className="flex items-center gap-1">
                  <a
                    href={p.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="text-gray-400 hover:text-blue-600 transition-colors"
                  >
                    <ExternalLink size={14} />
                  </a>
                  {onDelete && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Delete "${p.title.slice(0, 60)}…"?`)) {
                          onDelete(p.arxiv_id);
                        }
                      }}
                      className="text-gray-400 hover:text-red-600 transition-colors"
                      title="Delete paper"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
