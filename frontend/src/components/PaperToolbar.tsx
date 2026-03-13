"use client";

import { useState } from "react";
import { Search, LayoutGrid, List, SlidersHorizontal, Loader2 } from "lucide-react";

interface RefilterStatus {
  status: string;
  total: number;
  processed: number;
  removed: number;
}

interface Props {
  search: string;
  onSearchChange: (v: string) => void;
  venue: string;
  onVenueChange: (v: string) => void;
  dateFrom: string;
  onDateFromChange: (v: string) => void;
  dateTo: string;
  onDateToChange: (v: string) => void;
  view: "card" | "table";
  onViewChange: (v: "card" | "table") => void;
  onRefilter?: (body: { custom_instructions: string; min_quality: number; auto_delete: boolean }) => void;
  refilterStatus?: RefilterStatus | null;
}

export function PaperToolbar({
  search,
  onSearchChange,
  venue,
  onVenueChange,
  dateFrom,
  onDateFromChange,
  dateTo,
  onDateToChange,
  view,
  onViewChange,
  onRefilter,
  refilterStatus,
}: Props) {
  const [showRefilter, setShowRefilter] = useState(false);
  const [instructions, setInstructions] = useState("");
  const [minQuality, setMinQuality] = useState(3);
  const [autoDelete, setAutoDelete] = useState(false);

  const isRunning = refilterStatus?.status === "running";

  return (
    <div className="space-y-3 mb-4">
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
          />
          <input
            type="text"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search papers by title, abstract, or authors..."
            className="input pl-9"
          />
        </div>
        <input
          type="text"
          value={venue}
          onChange={(e) => onVenueChange(e.target.value)}
          placeholder="Filter by venue..."
          className="input w-full sm:w-44"
        />
        <div className="flex items-center gap-1.5">
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => onDateFromChange(e.target.value)}
            className="input w-[130px] text-sm py-1.5"
            title="Published from"
          />
          <span className="text-gray-400 text-xs">-</span>
          <input
            type="date"
            value={dateTo}
            onChange={(e) => onDateToChange(e.target.value)}
            className="input w-[130px] text-sm py-1.5"
            title="Published to"
          />
        </div>
        <div className="flex gap-2 self-start">
          {onRefilter && (
            <button
              onClick={() => setShowRefilter(!showRefilter)}
              className={`p-2 rounded-lg border transition-colors ${
                showRefilter
                  ? "bg-blue-50 text-blue-600 border-blue-300"
                  : "border-gray-300 hover:bg-gray-50 text-gray-500"
              }`}
              title="Re-filter papers"
            >
              <SlidersHorizontal size={16} />
            </button>
          )}
          <div className="flex rounded-lg border border-gray-300 overflow-hidden">
            <button
              onClick={() => onViewChange("card")}
              className={`p-2 transition-colors ${
                view === "card"
                  ? "bg-blue-50 text-blue-600"
                  : "hover:bg-gray-50 text-gray-500"
              }`}
              title="Card view"
            >
              <LayoutGrid size={16} />
            </button>
            <button
              onClick={() => onViewChange("table")}
              className={`p-2 transition-colors border-l border-gray-300 ${
                view === "table"
                  ? "bg-blue-50 text-blue-600"
                  : "hover:bg-gray-50 text-gray-500"
              }`}
              title="Table view"
            >
              <List size={16} />
            </button>
          </div>
        </div>
      </div>

      {showRefilter && onRefilter && (
        <div className="bg-gray-50 rounded-xl border border-gray-200 p-4 space-y-3">
          <h4 className="text-sm font-medium text-gray-700">Re-filter Papers</h4>

          <textarea
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            placeholder="Custom filter instructions (e.g., 'Remove all NLP papers', 'Keep only papers about diffusion models')..."
            className="input w-full h-20 resize-none text-sm"
          />

          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-gray-600">
              Min quality:
              <select
                value={minQuality}
                onChange={(e) => setMinQuality(Number(e.target.value))}
                className="input w-16 py-1 text-sm"
              >
                {[1, 2, 3, 4, 5].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>

            <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
              <input
                type="checkbox"
                checked={autoDelete}
                onChange={(e) => setAutoDelete(e.target.checked)}
                className="rounded border-gray-300"
              />
              Auto-delete below threshold
            </label>

            <button
              onClick={() => onRefilter({ custom_instructions: instructions, min_quality: minQuality, auto_delete: autoDelete })}
              disabled={isRunning}
              className="btn btn-primary text-sm px-4 py-1.5 ml-auto disabled:opacity-50"
            >
              {isRunning ? (
                <span className="flex items-center gap-2">
                  <Loader2 size={14} className="animate-spin" />
                  Running...
                </span>
              ) : (
                "Run Re-filter"
              )}
            </button>
          </div>

          {refilterStatus && refilterStatus.status !== "idle" && (
            <div className={`text-xs px-3 py-2 rounded-lg ${
              refilterStatus.status === "completed"
                ? "bg-emerald-50 text-emerald-700"
                : refilterStatus.status === "failed"
                ? "bg-red-50 text-red-700"
                : "bg-blue-50 text-blue-700"
            }`}>
              {refilterStatus.status === "running" && (
                <>Processing {refilterStatus.total} papers...</>
              )}
              {refilterStatus.status === "completed" && (
                <>Re-filter complete. {refilterStatus.processed} papers scored.{refilterStatus.removed > 0 && ` ${refilterStatus.removed} papers removed.`}</>
              )}
              {refilterStatus.status === "failed" && (
                <>Re-filter failed. Please try again.</>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
