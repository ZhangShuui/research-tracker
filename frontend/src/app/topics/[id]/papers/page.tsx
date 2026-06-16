"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, Paper } from "@/lib/api";
import { useDebounce } from "@/lib/hooks";
import { PaperToolbar } from "@/components/PaperToolbar";
import { PaperCard } from "@/components/PaperCard";
import { PaperTable } from "@/components/PaperTable";
import { PaperDetailModal } from "@/components/PaperDetailModal";
import { AddPaperModal } from "@/components/AddPaperModal";
import { GenerateInsightsModal } from "@/components/GenerateInsightsModal";
import { Pagination } from "@/components/Pagination";
import { Sparkles, CheckSquare, Loader2 } from "lucide-react";

const PAGE_SIZE = 24;

export default function PapersPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();

  const [search, setSearch] = useState("");
  const [venue, setVenue] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [view, setView] = useState<"card" | "table">("card");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Paper | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [picked, setPicked] = useState<Paper[]>([]);
  const [showGen, setShowGen] = useState(false);
  const pickedIds = new Set(picked.map((p) => p.arxiv_id));
  const togglePick = useCallback((paper: Paper) => {
    setPicked((prev) =>
      prev.some((p) => p.arxiv_id === paper.arxiv_id)
        ? prev.filter((p) => p.arxiv_id !== paper.arxiv_id)
        : [...prev, paper]
    );
  }, []);
  const [selectingAll, setSelectingAll] = useState(false);
  const [selectAllNote, setSelectAllNote] = useState("");
  const [refilterStatus, setRefilterStatus] = useState<{
    status: string;
    total: number;
    processed: number;
    removed: number;
  } | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const debouncedSearch = useDebounce(search, 300);
  const debouncedVenue = useDebounce(venue, 300);

  const selectAll = useCallback(async () => {
    setSelectingAll(true);
    setSelectAllNote("");
    try {
      // Grab everything matching the current filters (backend caps at 200).
      const res = await api.getPapers(id, {
        search: debouncedSearch, venue: debouncedVenue,
        date_from: dateFrom, date_to: dateTo, limit: 200, offset: 0,
      });
      setPicked(res.papers);
      if (res.total > res.papers.length) {
        setSelectAllNote(`selected ${res.papers.length} of ${res.total} (max 200) — narrow with filters for the rest`);
      }
    } catch {
      setSelectAllNote("Select all failed — try again.");
    } finally {
      setSelectingAll(false);
    }
  }, [id, debouncedSearch, debouncedVenue, dateFrom, dateTo]);

  const { data, isLoading } = useQuery({
    queryKey: ["papers", id, debouncedSearch, debouncedVenue, dateFrom, dateTo, offset],
    queryFn: () =>
      api.getPapers(id, {
        search: debouncedSearch,
        venue: debouncedVenue,
        date_from: dateFrom,
        date_to: dateTo,
        limit: PAGE_SIZE,
        offset,
      }),
    placeholderData: (prev) => prev,
  });

  // Reset offset when search changes
  const handleSearchChange = (v: string) => {
    setSearch(v);
    setOffset(0);
  };
  const handleVenueChange = (v: string) => {
    setVenue(v);
    setOffset(0);
  };
  const handleDateFromChange = (v: string) => {
    setDateFrom(v);
    setOffset(0);
  };
  const handleDateToChange = (v: string) => {
    setDateTo(v);
    setOffset(0);
  };

  const invalidatePapers = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["papers", id] });
  }, [queryClient, id]);

  const handleAdded = useCallback((summarizing: boolean) => {
    setShowAdd(false);
    setOffset(0); // newest paper appears first (added_at DESC)
    invalidatePapers();
    if (summarizing) {
      // background summarize fills the structured fields a bit later
      setTimeout(() => invalidatePapers(), 20000);
    }
  }, [invalidatePapers]);

  const handleDelete = useCallback(async (arxivId: string) => {
    try {
      await api.deletePaper(id, arxivId);
      invalidatePapers();
      if (selected?.arxiv_id === arxivId) {
        setSelected(null);
      }
    } catch (e) {
      console.error("Failed to delete paper:", e);
    }
  }, [id, invalidatePapers, selected]);

  const [backfillStatus, setBackfillStatus] = useState<{
    status: string;
    total: number;
    processed: number;
  } | null>(null);
  const backfillPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (backfillPollRef.current) clearInterval(backfillPollRef.current);
    };
  }, []);

  const handleBackfill = useCallback(async () => {
    try {
      const res = await api.backfillSummaries(id);
      if (res.status === "completed") {
        // nothing was missing — no background job
        setBackfillStatus({ status: "completed", total: res.total ?? 0, processed: res.total ?? 0 });
        return;
      }
      setBackfillStatus({ status: "running", total: res.total ?? 0, processed: 0 });
      if (backfillPollRef.current) clearInterval(backfillPollRef.current);
      backfillPollRef.current = setInterval(async () => {
        try {
          const status = await api.getBackfillStatus(id);
          setBackfillStatus(status);
          if (status.status === "completed" || status.status === "failed") {
            if (backfillPollRef.current) clearInterval(backfillPollRef.current);
            backfillPollRef.current = null;
            invalidatePapers();
          }
        } catch {
          if (backfillPollRef.current) clearInterval(backfillPollRef.current);
          backfillPollRef.current = null;
        }
      }, 2000);
    } catch (e) {
      console.error("Failed to start backfill:", e);
      setBackfillStatus({ status: "failed", total: 0, processed: 0 });
    }
  }, [id, invalidatePapers]);

  const handleRefilter = useCallback(async (body: {
    custom_instructions: string;
    min_quality: number;
    auto_delete: boolean;
  }) => {
    try {
      await api.refilterPapers(id, body);
      setRefilterStatus({ status: "running", total: 0, processed: 0, removed: 0 });

      // Poll for status
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getRefilterStatus(id);
          setRefilterStatus(status);
          if (status.status === "completed" || status.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            invalidatePapers();
          }
        } catch {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
        }
      }, 2000);
    } catch (e) {
      console.error("Failed to start refilter:", e);
    }
  }, [id, invalidatePapers]);

  const handleDeepRead = useCallback((arxivId: string) => {
    router.push(`/topics/${id}/deep-read?paper=${encodeURIComponent(arxivId)}`);
  }, [id, router]);

  return (
    <div className="space-y-4">
      <PaperToolbar
        search={search}
        onSearchChange={handleSearchChange}
        venue={venue}
        onVenueChange={handleVenueChange}
        dateFrom={dateFrom}
        onDateFromChange={handleDateFromChange}
        dateTo={dateTo}
        onDateToChange={handleDateToChange}
        view={view}
        onViewChange={setView}
        onAdd={() => setShowAdd(true)}
        onRefilter={handleRefilter}
        refilterStatus={refilterStatus}
        onBackfill={handleBackfill}
        backfillStatus={backfillStatus}
      />

      {data && data.papers.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500 -mt-1 px-1">
          <button
            onClick={selectAll}
            disabled={selectingAll}
            className="inline-flex items-center gap-1.5 text-indigo-600 hover:text-indigo-800 font-medium disabled:opacity-50"
          >
            {selectingAll ? <Loader2 size={13} className="animate-spin" /> : <CheckSquare size={13} />}
            {selectingAll ? "Selecting…" : `Select all${data.total ? ` (${data.total})` : ""}`}
          </button>
          {picked.length > 0 && (
            <>
              <span className="text-gray-300">·</span>
              <span className="text-gray-600 font-medium">{picked.length} selected</span>
              <button onClick={() => { setPicked([]); setSelectAllNote(""); }} className="text-gray-400 hover:text-gray-600">
                Clear
              </button>
            </>
          )}
          {selectAllNote && <span className="text-amber-600">{selectAllNote}</span>}
        </div>
      )}

      {isLoading && !data ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="bg-white rounded-xl border border-gray-200 p-4 h-44 animate-pulse"
            />
          ))}
        </div>
      ) : !data?.papers.length ? (
        <div className="text-center py-16">
          <p className="text-gray-500">
            {debouncedSearch || debouncedVenue || dateFrom || dateTo
              ? "No papers match your filters."
              : "No papers yet. Run the pipeline to discover papers."}
          </p>
        </div>
      ) : view === "card" ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.papers.map((p) => (
            <PaperCard
              key={p.arxiv_id}
              paper={p}
              onClick={() => setSelected(p)}
              onDelete={handleDelete}
              onDeepRead={handleDeepRead}
              selected={pickedIds.has(p.arxiv_id)}
              onToggleSelect={togglePick}
            />
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <PaperTable
            papers={data.papers}
            onSelect={setSelected}
            onDelete={handleDelete}
            onDeepRead={handleDeepRead}
            selectedIds={pickedIds}
            onToggleSelect={togglePick}
          />
        </div>
      )}

      {data && (
        <Pagination
          total={data.total}
          limit={data.limit}
          offset={offset}
          onChange={setOffset}
        />
      )}

      {selected && (
        <PaperDetailModal
          paper={selected}
          onClose={() => setSelected(null)}
          onDelete={handleDelete}
          onDeepRead={handleDeepRead}
        />
      )}

      {showAdd && (
        <AddPaperModal
          topicId={id}
          onClose={() => setShowAdd(false)}
          onAdded={handleAdded}
        />
      )}

      {/* Selection action bar */}
      {picked.length > 0 && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 flex items-center gap-3 bg-gray-900 text-white rounded-full shadow-xl pl-5 pr-3 py-2.5">
          <span className="text-sm font-medium tabular-nums">{picked.length} selected</span>
          <button
            onClick={() => setShowGen(true)}
            className="inline-flex items-center gap-1.5 text-sm bg-indigo-500 hover:bg-indigo-400 rounded-full px-3.5 py-1.5 font-medium transition-colors"
          >
            <Sparkles size={14} />
            Generate insights
          </button>
          <button
            onClick={() => setPicked([])}
            className="text-xs text-gray-300 hover:text-white px-1.5"
          >
            Clear
          </button>
        </div>
      )}

      {showGen && (
        <GenerateInsightsModal
          topicId={id}
          papers={picked}
          onClose={() => setShowGen(false)}
          onGenerated={() => {
            setShowGen(false);
            setPicked([]);
            router.push(`/topics/${id}/insights`);
          }}
        />
      )}
    </div>
  );
}
