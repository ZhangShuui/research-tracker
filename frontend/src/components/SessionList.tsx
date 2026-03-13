"use client";

import { Session } from "@/lib/api";
import { FileText, GitFork } from "lucide-react";

interface Props {
  sessions: Session[];
  onSelect: (session: Session) => void;
  selectedId?: string;
}

export function SessionList({ sessions, onSelect, selectedId }: Props) {
  if (sessions.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-slate-100 flex items-center justify-center">
          <FileText size={20} className="text-slate-300" />
        </div>
        <p className="text-sm text-slate-500 font-medium">No sessions yet</p>
        <p className="text-xs text-slate-400 mt-1">
          Trigger a run to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200/60">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-50/80">
            <th className="text-left text-xs text-slate-500 font-medium px-4 py-2.5">
              Session
            </th>
            <th className="text-left text-xs text-slate-500 font-medium px-4 py-2.5">
              Started
            </th>
            <th className="text-center text-xs text-slate-500 font-medium px-4 py-2.5">
              Papers
            </th>
            <th className="text-center text-xs text-slate-500 font-medium px-4 py-2.5">
              Repos
            </th>
            <th className="text-left text-xs text-slate-500 font-medium px-4 py-2.5">
              Status
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {sessions.map((s, idx) => (
            <tr
              key={s.id}
              onClick={() => onSelect(s)}
              className={`cursor-pointer transition-colors duration-150 ${
                selectedId === s.id
                  ? "bg-indigo-50/70 hover:bg-indigo-50"
                  : idx % 2 === 0
                  ? "bg-white hover:bg-slate-50"
                  : "bg-slate-50/30 hover:bg-slate-50"
              }`}
            >
              <td className="px-4 py-3 font-mono text-xs text-indigo-600 font-medium">
                {s.id}
              </td>
              <td className="px-4 py-3 text-slate-600 text-xs">
                {s.started_at
                  ? new Date(s.started_at).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })
                  : "\u2014"}
              </td>
              <td className="px-4 py-3 text-center">
                <span className="inline-flex items-center gap-1 text-xs text-slate-600">
                  <FileText size={11} className="text-indigo-300" />
                  <span className="font-medium tabular-nums">
                    {s.paper_count}
                  </span>
                </span>
              </td>
              <td className="px-4 py-3 text-center">
                <span className="inline-flex items-center gap-1 text-xs text-slate-600">
                  <GitFork size={11} className="text-emerald-300" />
                  <span className="font-medium tabular-nums">
                    {s.repo_count}
                  </span>
                </span>
              </td>
              <td className="px-4 py-3">
                <StatusBadge status={s.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<
    string,
    { bg: string; dot: string; text: string }
  > = {
    running: {
      bg: "bg-amber-50 border-amber-200/60",
      dot: "bg-amber-400 animate-pulse",
      text: "text-amber-700",
    },
    completed: {
      bg: "bg-emerald-50 border-emerald-200/60",
      dot: "bg-emerald-400",
      text: "text-emerald-700",
    },
    failed: {
      bg: "bg-red-50 border-red-200/60",
      dot: "bg-red-400",
      text: "text-red-600",
    },
  };
  const style = map[status] ?? {
    bg: "bg-slate-50 border-slate-200",
    dot: "bg-slate-400",
    text: "text-slate-600",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[11px] px-2.5 py-1 rounded-full font-medium border ${style.bg} ${style.text}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
      {status}
    </span>
  );
}
