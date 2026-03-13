"use client";

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { api, Session } from "@/lib/api";
import { TranslateButton, TranslatedBlock } from "./TranslateButton";
import { MathMarkdown } from "./MathMarkdown";

interface Props {
  topicId: string;
  session: Session;
  onClose: () => void;
}

const SECTION_HEADINGS = [
  "Executive Summary",
  "Thematic Analysis",
  "Paper Details",
] as const;

function splitReportSections(content: string): Record<string, string> {
  const sections: Record<string, string> = {};
  const lines = content.split("\n");
  let currentKey = "full";
  let currentLines: string[] = [];

  for (const line of lines) {
    const heading = SECTION_HEADINGS.find(
      (h) => line.startsWith(`## ${h}`) || line.startsWith(`## ${h} (`)
    );
    if (heading) {
      if (currentLines.length) {
        sections[currentKey] = currentLines.join("\n").trim();
      }
      currentKey = heading;
      currentLines = [];
    } else {
      currentLines.push(line);
    }
  }
  if (currentLines.length) {
    sections[currentKey] = currentLines.join("\n").trim();
  }
  return sections;
}

export function ReportViewer({ topicId, session, onClose }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ["session", topicId, session.id],
    queryFn: () => api.getSession(topicId, session.id),
    initialData: session.report_content !== undefined ? session : undefined,
  });

  const sections = useMemo(() => {
    if (!data?.report_content) return null;
    const parsed = splitReportSections(data.report_content);
    // If we found at least one named section, it's the new format
    if (SECTION_HEADINGS.some((h) => h in parsed)) return parsed;
    return null;
  }, [data?.report_content]);

  const [activeSection, setActiveSection] = useState<string>("Executive Summary");
  const [translatedText, setTranslatedText] = useState<string | null>(null);

  // Available tabs from parsed sections
  const availableTabs = sections
    ? SECTION_HEADINGS.filter((h) => h in sections)
    : [];

  return (
    <div className="fixed inset-0 bg-black/40 flex items-end sm:items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-3xl max-h-[85vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b flex-shrink-0">
          <div>
            <h2 className="font-semibold text-base">Session Report</h2>
            <p className="text-xs text-gray-500 font-mono">{session.id}</p>
          </div>
          <div className="flex items-center gap-1.5">
            {data?.report_content && (
              <TranslateButton
                sourceType="report"
                sourceId={session.id}
                field={sections ? activeSection : "full"}
                content={
                  sections && activeSection in sections
                    ? sections[activeSection]
                    : data.report_content
                }
                onTranslated={setTranslatedText}
              />
            )}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Section tabs — only for new-format reports */}
        {availableTabs.length > 0 && (
          <div className="flex gap-1 px-4 pt-2 border-b border-gray-100 overflow-x-auto">
            {availableTabs.map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveSection(tab)}
                className={`px-3 py-2 text-xs font-medium whitespace-nowrap border-b-2 transition-colors ${
                  activeSection === tab
                    ? "border-blue-600 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        )}

        <div className="overflow-y-auto flex-1 p-5">
          {isLoading ? (
            <Spinner />
          ) : data?.report_content ? (
            <>
              {sections && activeSection in sections ? (
                <Prose content={sections[activeSection]} />
              ) : (
                // Fallback: render full markdown for old-format reports
                <Prose content={data.report_content} />
              )}
              {translatedText && <TranslatedBlock content={translatedText} />}
            </>
          ) : (
            <p className="text-sm text-gray-500 text-center py-8">
              No report available for this session.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function Prose({ content }: { content: string }) {
  return (
    <MathMarkdown className="prose prose-sm max-w-none">{content}</MathMarkdown>
  );
}

function Spinner() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
    </div>
  );
}
