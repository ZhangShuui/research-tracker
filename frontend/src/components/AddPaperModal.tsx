"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { X, Loader2, Download } from "lucide-react";
import { api, AddPaperBody } from "@/lib/api";
import { BatchAddPanel } from "./BatchAddPanel";

interface Props {
  topicId: string;
  onClose: () => void;
  /** Called after a successful add. `summarizing` is true if a background
   *  summarize job was kicked off (parent re-fetches once it likely finished). */
  onAdded: (summarizing: boolean) => void;
}

function parseErr(e: unknown): { status: number; detail: string } {
  const msg = e instanceof Error ? e.message : String(e);
  const m = msg.match(/^(\d{3})\s+([\s\S]*)$/);
  const status = m ? parseInt(m[1], 10) : 0;
  let detail = m ? m[2] : msg;
  try {
    const j = JSON.parse(detail);
    if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
  } catch {
    /* not JSON — keep raw text */
  }
  return { status, detail };
}

export function AddPaperModal({ topicId, onClose, onAdded }: Props) {
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [arxiv, setArxiv] = useState("");
  const [title, setTitle] = useState("");
  const [authors, setAuthors] = useState("");
  const [abstract, setAbstract] = useState("");
  const [url, setUrl] = useState("");
  const [published, setPublished] = useState("");
  const [venue, setVenue] = useState("");
  const [summarize, setSummarize] = useState(false);

  const [fetched, setFetched] = useState(false);
  const [fetchMsg, setFetchMsg] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const lookupMut = useMutation({
    mutationFn: (query: string) => api.lookupPaper(topicId, query),
    onSuccess: (res) => {
      if (res.found && res.paper) {
        const p = res.paper;
        setArxiv(p.arxiv_id);
        setTitle(p.title || "");
        setAuthors(p.authors || "");
        setAbstract(p.abstract || "");
        setUrl(p.url || "");
        setPublished(p.published || "");
        setVenue(p.venue || "");
        setFetched(true);
        setFetchMsg(null);
      } else {
        setFetched(false);
        setFetchMsg("No arXiv paper found for that ID/URL — you can still enter details manually below.");
      }
    },
    onError: (e) => {
      const { status, detail } = parseErr(e);
      setFetchMsg(status === 502
        ? "arXiv lookup failed (network). Try again, or enter details manually below."
        : detail || "Lookup failed.");
    },
  });

  const addMut = useMutation({
    mutationFn: (body: AddPaperBody) => api.addPaper(topicId, body),
    onSuccess: (res) => onAdded(res.summarizing),
    onError: (e) => {
      const { status, detail } = parseErr(e);
      if (status === 409) setSaveError("This paper is already in your library.");
      else setSaveError(detail || "Failed to add paper.");
    },
  });

  const canSave = title.trim().length > 0 && !addMut.isPending;

  const handleFetch = () => {
    if (!arxiv.trim() || lookupMut.isPending) return;
    setFetchMsg(null);
    lookupMut.mutate(arxiv.trim());
  };

  const handleSave = () => {
    if (!canSave) return;
    setSaveError(null);
    addMut.mutate({
      arxiv_id: arxiv.trim(),
      title: title.trim(),
      authors: authors.trim(),
      abstract: abstract.trim(),
      url: url.trim(),
      published: published.trim(),
      venue: venue.trim(),
      summarize,
    });
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-end sm:items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <h3 className="text-base font-semibold text-gray-900">Add paper</h3>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors text-gray-500"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        {/* Mode tabs */}
        <div className="flex gap-1 px-5 pt-3 border-b border-gray-100">
          <button
            onClick={() => setMode("single")}
            className={`px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              mode === "single"
                ? "border-indigo-600 text-indigo-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            Single paper
          </button>
          <button
            onClick={() => setMode("batch")}
            className={`px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              mode === "batch"
                ? "border-indigo-600 text-indigo-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            Paste URLs
          </button>
        </div>

        {mode === "batch" ? (
          <BatchAddPanel topicId={topicId} onClose={onClose} />
        ) : (
        <>
        {/* Body */}
        <div className="overflow-y-auto flex-1 p-5 space-y-4">
          {/* arXiv auto-fill */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              arXiv ID or URL <span className="font-normal text-gray-400">(optional)</span>
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={arxiv}
                onChange={(e) => setArxiv(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); handleFetch(); } }}
                placeholder="e.g. 2401.12345 or https://arxiv.org/abs/2401.12345"
                className="input flex-1"
              />
              <button
                onClick={handleFetch}
                disabled={!arxiv.trim() || lookupMut.isPending}
                className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-sm rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50 text-gray-700"
              >
                {lookupMut.isPending
                  ? <Loader2 size={15} className="animate-spin" />
                  : <Download size={15} />}
                Fetch
              </button>
            </div>
            <p className="mt-1 text-xs text-gray-400">
              Paste an arXiv link to auto-fill the fields, or leave blank to add any paper manually.
            </p>
            {fetched && (
              <p className="mt-1 text-xs text-emerald-600">Fetched from arXiv ✓ — review and edit below.</p>
            )}
            {fetchMsg && (
              <p className="mt-1 text-xs text-amber-600">{fetchMsg}</p>
            )}
          </div>

          {/* Title (required) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Title <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Paper title"
              className="input"
            />
          </div>

          {/* Authors */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Authors</label>
            <input
              type="text"
              value={authors}
              onChange={(e) => setAuthors(e.target.value)}
              placeholder="Comma-separated, e.g. Jane Doe, John Smith"
              className="input"
            />
          </div>

          {/* Abstract */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Abstract</label>
            <textarea
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
              placeholder="Paper abstract / summary"
              className="input h-28 resize-none"
            />
          </div>

          {/* URL + Published */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">URL</label>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://..."
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Published</label>
              <input
                type="date"
                value={published}
                onChange={(e) => setPublished(e.target.value)}
                className="input"
              />
            </div>
          </div>

          {/* Venue */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Venue</label>
            <input
              type="text"
              value={venue}
              onChange={(e) => setVenue(e.target.value)}
              placeholder="e.g. NeurIPS 2025, arXiv preprint"
              className="input"
            />
          </div>

          {/* Auto-summarize */}
          <label className="flex items-start gap-2.5 cursor-pointer rounded-lg border border-gray-200 p-3 hover:bg-gray-50">
            <input
              type="checkbox"
              checked={summarize}
              onChange={(e) => setSummarize(e.target.checked)}
              className="mt-0.5 rounded border-gray-300"
            />
            <span className="text-sm text-gray-700">
              Auto-summarize with AI after adding
              <span className="block text-xs text-gray-400">
                Extracts key insight, method &amp; contribution in the background. May take a little while.
              </span>
            </span>
          </label>
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-gray-100 space-y-3">
          {saveError && (
            <p className="text-sm text-red-600">{saveError}</p>
          )}
          <div className="flex justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={!canSave}
              className="inline-flex items-center gap-2 px-5 py-2.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 font-medium"
            >
              {addMut.isPending && <Loader2 size={15} className="animate-spin" />}
              {addMut.isPending ? "Adding..." : "Add paper"}
            </button>
          </div>
        </div>
        </>
        )}
      </div>
    </div>
  );
}
