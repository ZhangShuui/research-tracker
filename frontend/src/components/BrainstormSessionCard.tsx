"use client";

import { BrainstormSession } from "@/lib/api";
import { Lightbulb, Clock, CheckCircle2, XCircle } from "lucide-react";

interface Props {
  session: BrainstormSession;
  onClick: () => void;
  isSelected: boolean;
}

export function BrainstormSessionCard({ session, onClick, isSelected }: Props) {
  const statusIcon = {
    running: <Clock size={14} className="text-yellow-500 animate-pulse" />,
    completed: <CheckCircle2 size={14} className="text-green-500" />,
    failed: <XCircle size={14} className="text-red-500" />,
  }[session.status];

  const ideaCount = session.ideas_json?.length ?? 0;

  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-xl border p-4 transition-all ${
        isSelected
          ? "border-blue-300 bg-blue-50 shadow-sm"
          : "border-gray-200 bg-white hover:shadow-sm hover:border-gray-300"
      }`}
    >
      <div className="flex items-center gap-2 mb-1.5">
        {statusIcon}
        <span className="text-xs font-mono text-gray-500">{session.id}</span>
        <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 capitalize">
          {session.mode}
        </span>
      </div>

      {session.user_idea && (
        <p className="text-xs text-gray-600 line-clamp-1 mb-1.5">
          &ldquo;{session.user_idea}&rdquo;
        </p>
      )}

      <div className="flex items-center gap-3 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <Lightbulb size={12} />
          {ideaCount} idea{ideaCount !== 1 ? "s" : ""}
        </span>
        {session.started_at && (
          <span>{new Date(session.started_at).toLocaleString()}</span>
        )}
      </div>
    </button>
  );
}
