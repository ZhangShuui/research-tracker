"use client";

import { useState } from "react";
import { X, ExternalLink, Trash2 } from "lucide-react";
import { Paper } from "@/lib/api";
import { TranslateButton, TranslatedBlock } from "./TranslateButton";

interface Props {
  paper: Paper;
  onClose: () => void;
  onDelete?: (arxivId: string) => void;
}

export function PaperDetailModal({ paper, onClose, onDelete }: Props) {
  const [translatedText, setTranslatedText] = useState<string | null>(null);

  const fullContent = [
    paper.key_insight && `**Key Insight:** ${paper.key_insight}`,
    paper.method && `**Method:** ${paper.method}`,
    paper.contribution && `**Contribution:** ${paper.contribution}`,
    paper.summary && `**Summary:** ${paper.summary}`,
  ]
    .filter(Boolean)
    .join("\n\n");

  return (
    <div className="fixed inset-0 bg-black/40 flex items-end sm:items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[85vh] flex flex-col">
        <div className="flex items-start justify-between p-5 border-b flex-shrink-0 gap-3">
          <div className="min-w-0">
            <h2 className="font-semibold text-base leading-snug">
              {paper.title}
            </h2>
            <p className="text-xs text-gray-500 mt-1">
              {paper.authors}
            </p>
          </div>
          <div className="flex items-center gap-1 flex-shrink-0">
            <a
              href={paper.url}
              target="_blank"
              rel="noopener noreferrer"
              className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors text-gray-500"
              title="Open on arXiv"
            >
              <ExternalLink size={16} />
            </a>
            <TranslateButton
              sourceType="paper"
              sourceId={paper.arxiv_id}
              field="full"
              content={fullContent}
              onTranslated={setTranslatedText}
            />
            {onDelete && (
              <button
                onClick={() => {
                  if (confirm(`Delete "${paper.title.slice(0, 60)}…"?`)) {
                    onDelete(paper.arxiv_id);
                  }
                }}
                className="p-1.5 rounded-lg hover:bg-red-50 hover:text-red-600 transition-colors text-gray-500"
                title="Delete paper"
              >
                <Trash2 size={16} />
              </button>
            )}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="overflow-y-auto flex-1 p-5 space-y-4">
          {/* Metadata row */}
          <div className="flex flex-wrap gap-2">
            {paper.quality_score > 0 && (
              <span
                className={`text-xs px-2.5 py-1 rounded-full font-medium ${
                  paper.quality_score >= 4
                    ? "bg-emerald-50 text-emerald-700"
                    : paper.quality_score >= 3
                    ? "bg-amber-50 text-amber-700"
                    : "bg-red-50 text-red-700"
                }`}
              >
                Quality: {paper.quality_score}/5
              </span>
            )}
            {paper.venue && (
              <span className="text-xs px-2.5 py-1 rounded-full bg-green-50 text-green-700 font-medium">
                {paper.venue}
              </span>
            )}
            <span className="text-xs px-2.5 py-1 rounded-full bg-gray-100 text-gray-600">
              {new Date(paper.published).toLocaleDateString()}
            </span>
            <span className="text-xs px-2.5 py-1 rounded-full bg-gray-100 text-gray-600 font-mono">
              {paper.arxiv_id}
            </span>
          </div>

          {/* Key fields */}
          {paper.key_insight && (
            <Section title="Key Insight">
              <p className="text-sm text-blue-800 bg-blue-50 rounded-lg p-3">
                {paper.key_insight}
              </p>
            </Section>
          )}

          {paper.method && (
            <Section title="Method">
              <p className="text-sm text-gray-700">{paper.method}</p>
            </Section>
          )}

          {paper.contribution && (
            <Section title="Contribution">
              <p className="text-sm text-gray-700">{paper.contribution}</p>
            </Section>
          )}

          <Section title="Summary">
            <p className="text-sm text-gray-700 leading-relaxed">
              {paper.summary}
            </p>
          </Section>

          {paper.math_concepts?.length > 0 && (
            <Section title="Mathematical Concepts">
              <div className="flex flex-wrap gap-2">
                {paper.math_concepts.map((mc) => (
                  <span
                    key={mc}
                    className="text-xs px-2.5 py-1 rounded-full bg-purple-50 text-purple-700"
                  >
                    {mc}
                  </span>
                ))}
              </div>
            </Section>
          )}

          {paper.cited_works?.length > 0 && (
            <Section title="Notable Cited Works">
              <ul className="text-sm text-gray-700 space-y-1 list-disc list-inside">
                {paper.cited_works.map((cw) => (
                  <li key={cw}>{cw}</li>
                ))}
              </ul>
            </Section>
          )}

          {translatedText && (
            <Section title="中文翻译">
              <TranslatedBlock content={translatedText} />
            </Section>
          )}

          <Section title="Abstract">
            <p className="text-sm text-gray-600 leading-relaxed">
              {paper.abstract}
            </p>
          </Section>
        </div>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
        {title}
      </h3>
      {children}
    </div>
  );
}
