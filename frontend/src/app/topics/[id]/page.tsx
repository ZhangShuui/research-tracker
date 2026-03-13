"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import {
  Search,
  Tag,
  CalendarDays,
  Clock,
  FileText,
  GitFork,
} from "lucide-react";
import { api, Session } from "@/lib/api";
import { SessionList } from "@/components/SessionList";
import { ReportViewer } from "@/components/ReportViewer";

export default function TopicOverviewPage() {
  const { id } = useParams<{ id: string }>();
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);

  const { data: topic } = useQuery({
    queryKey: ["topic", id],
    queryFn: () => api.getTopic(id),
    refetchInterval: 10_000,
  });

  const { data: sessionsData } = useQuery({
    queryKey: ["sessions", id],
    queryFn: () => api.getSessions(id),
    enabled: Boolean(topic),
    refetchInterval: 10_000,
  });

  if (!topic) return null;

  const totalPapers =
    sessionsData?.sessions.reduce((sum, s) => sum + s.paper_count, 0) ?? 0;
  const totalRepos =
    sessionsData?.sessions.reduce((sum, s) => sum + s.repo_count, 0) ?? 0;

  return (
    <div className="space-y-6">
      {/* Config summary grid with icons */}
      <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200/60 shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-100 bg-slate-50/50">
          <h2 className="text-sm font-semibold text-slate-700">
            Configuration
          </h2>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-0 divide-x divide-slate-100">
          <ConfigItem
            icon={<Search size={15} className="text-indigo-400" />}
            label="arXiv Keywords"
            value={topic.arxiv_keywords.join(", ")}
          />
          <ConfigItem
            icon={<Tag size={15} className="text-violet-400" />}
            label="Categories"
            value={topic.arxiv_categories.join(", ")}
          />
          <ConfigItem
            icon={<CalendarDays size={15} className="text-blue-400" />}
            label="arXiv Lookback"
            value={`${topic.arxiv_lookback_days} days`}
          />
          <ConfigItem
            icon={<Clock size={15} className="text-amber-400" />}
            label="Schedule"
            value={topic.schedule_cron || "Manual only"}
            mono={Boolean(topic.schedule_cron)}
          />
        </div>
        {/* Aggregate stats bar */}
        <div className="px-5 py-3 border-t border-slate-100 bg-slate-50/30 flex items-center gap-6 text-xs">
          <div className="flex items-center gap-1.5 text-slate-500">
            <FileText size={13} className="text-indigo-400" />
            <span className="font-semibold text-slate-700 tabular-nums">
              {totalPapers}
            </span>{" "}
            papers across all sessions
          </div>
          <div className="flex items-center gap-1.5 text-slate-500">
            <GitFork size={13} className="text-emerald-400" />
            <span className="font-semibold text-slate-700 tabular-nums">
              {totalRepos}
            </span>{" "}
            repos tracked
          </div>
        </div>
      </div>

      {/* Sessions */}
      <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200/60 shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-700">
            Sessions
          </h2>
          <span className="text-xs text-slate-400 font-medium tabular-nums">
            {sessionsData?.sessions.length ?? 0} total
          </span>
        </div>
        <div className="p-5">
          <SessionList
            sessions={sessionsData?.sessions ?? []}
            onSelect={setSelectedSession}
            selectedId={selectedSession?.id}
          />
        </div>
      </div>

      {/* Report drawer */}
      {selectedSession && (
        <ReportViewer
          topicId={id}
          session={selectedSession}
          onClose={() => setSelectedSession(null)}
        />
      )}
    </div>
  );
}

function ConfigItem({
  icon,
  label,
  value,
  mono,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="p-4 hover:bg-slate-50/50 transition-colors">
      <div className="flex items-center gap-2 mb-1.5">
        {icon}
        <p className="text-xs text-slate-400 font-medium">{label}</p>
      </div>
      <p
        className={`text-sm text-slate-700 truncate ${
          mono ? "font-mono text-xs bg-slate-100 px-2 py-0.5 rounded inline-block" : "font-medium"
        }`}
        title={value}
      >
        {value || "\u2014"}
      </p>
    </div>
  );
}
