"use client";

import { MathMarkdown } from "./MathMarkdown";

interface Props {
  literatureResult: string;
  logicResult: string;
  codeResult: string;
}

export function VerificationResults({
  literatureResult,
  logicResult,
  codeResult,
}: Props) {
  const hasAny = literatureResult || logicResult || codeResult;
  if (!hasAny) return null;

  return (
    <div className="space-y-4">
      {literatureResult && (
        <ResultSection
          title="Literature Verification"
          color="blue"
          content={literatureResult}
        />
      )}
      {logicResult && (
        <ResultSection
          title="Logic & Feasibility Analysis"
          color="amber"
          content={logicResult}
        />
      )}
      {codeResult && (
        <ResultSection
          title="Code Proof-of-Concept"
          color="green"
          content={codeResult}
          code
        />
      )}
    </div>
  );
}

function ResultSection({
  title,
  color,
  content,
  code,
}: {
  title: string;
  color: string;
  content: string;
  code?: boolean;
}) {
  const bgMap: Record<string, string> = {
    blue: "bg-blue-50 border-blue-100",
    amber: "bg-amber-50 border-amber-100",
    green: "bg-green-50 border-green-100",
  };
  const titleMap: Record<string, string> = {
    blue: "text-blue-900",
    amber: "text-amber-900",
    green: "text-green-900",
  };

  return (
    <div className={`rounded-xl border p-4 ${bgMap[color] || "bg-gray-50 border-gray-100"}`}>
      <h3 className={`font-semibold text-sm mb-3 ${titleMap[color] || "text-gray-900"}`}>
        {title}
      </h3>
      {code ? (
        <pre className="text-xs bg-gray-900 text-green-400 rounded-lg p-4 overflow-x-auto whitespace-pre-wrap">
          {content}
        </pre>
      ) : (
        <MathMarkdown className="prose prose-sm max-w-none">{content}</MathMarkdown>
      )}
    </div>
  );
}
