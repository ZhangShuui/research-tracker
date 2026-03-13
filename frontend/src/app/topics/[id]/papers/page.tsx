"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useParams } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, Paper } from "@/lib/api";
import { useDebounce } from "@/lib/hooks";
import { PaperToolbar } from "@/components/PaperToolbar";
import { PaperCard } from "@/components/PaperCard";
import { PaperTable } from "@/components/PaperTable";
import { PaperDetailModal } from "@/components/PaperDetailModal";
import { Pagination } from "@/components/Pagination";

const PAGE_SIZE = 24;

export default function PapersPage() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();

  const [search, setSearch] = useState("");
  const [venue, setVenue] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [view, setView] = useState<"card" | "table">("card");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Paper | null>(null);
  const [refilterStatus, setRefilterStatus] = useState<{
    status: string;
    total: number;
    processed: number;
    removed: number;
  } | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const debouncedSearch = useDebounce(search, 300);
  const debouncedVenue = useDebounce(venue, 300);

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

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

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
        onRefilter={handleRefilter}
        refilterStatus={refilterStatus}
      />

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
            />
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <PaperTable
            papers={data.papers}
            onSelect={setSelected}
            onDelete={handleDelete}
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
        />
      )}
    </div>
  );
}
