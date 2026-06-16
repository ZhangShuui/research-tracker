"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Loader2, Check, AlertCircle, MinusCircle } from "lucide-react";
import { api, BatchAddResponse, BatchAddResultItem } from "@/lib/api";

interface Props {
  topicId: string;
  onClose: () => void;
}

const STATUS_META: Record<
  BatchAddResultItem["status"],
  { label: string; cls: string; icon: typeof Check }
> = {
  added: { label: "Added", cls: "text-emerald-600", icon: Check },
  duplicate: { label: "Already in library", cls: "text-gray-400", icon: MinusCircle },
  not_found: { label: "Not found on arXiv", cls: "text-amber-600", icon: AlertCircle },
  invalid: { label: "Not an arXiv link/ID", cls: "text-amber-600", icon: AlertCircle },
  error: { label: "Fetch failed", cls: "text-red-600", icon: AlertCircle },
};

export function BatchAddPanel({ topicId, onClose }: Props) {
  const qc = useQueryClient();
  const [text, setText] = useState("");
  const [summarize, setSummarize] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BatchAddResponse | null>(null);

  const handleAdd = async () => {
    if (!text.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.batchAddPapers(topicId, { text, summarize });
      setResult(res);
      if ((res.counts.added ?? 0) > 0) {
        qc.invalidateQueries({ queryKey: ["papers", topicId] });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch add failed.");
    } finally {
      setLoading(false);
    }
  };

  const c = result?.counts;

  return (
    <>
      <div className="overflow-y-auto flex-1 p-5 space-y-3">
        <label className="block text-sm font-medium text-gray-700">
          arXiv URLs or IDs
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={"Paste arXiv links or IDs, separated by commas or new lines:\n\n2401.12345, https://arxiv.org/abs/2402.06789\narXiv:2403.00001"}
          className="input h-36 resize-none font-mono text-xs leading-relaxed"
        />
        <p className="text-xs text-gray-400">
          arXiv links/IDs only — other URLs can&apos;t be auto-fetched and will be reported as skipped.
          All fetched in a single request.
        </p>

        <label className="flex items-start gap-2.5 cursor-pointer rounded-lg border border-gray-200 p-3 hover:bg-gray-50">
          <input
            type="checkbox"
            checked={summarize}
            onChange={(e) => setSummarize(e.target.checked)}
            className="mt-0.5 rounded border-gray-300"
          />
          <span className="text-sm text-gray-700">
            Auto-summarize all added papers with AI
            <span className="block text-xs text-gray-400">
              Runs in the background after adding. May take a while for many papers.
            </span>
          </span>
        </label>

        {error && <p className="text-sm text-red-600">{error}</p>}

        {result && (
          <div className="space-y-2 pt-1">
            <div className="flex flex-wrap gap-3 text-xs">
              <span className="text-emerald-600 font-medium">{c?.added ?? 0} added</span>
              {(c?.duplicate ?? 0) > 0 && <span className="text-gray-400">{c?.duplicate} duplicate</span>}
              {(c?.not_found ?? 0) > 0 && <span className="text-amber-600">{c?.not_found} not found</span>}
              {(c?.invalid ?? 0) > 0 && <span className="text-amber-600">{c?.invalid} not arXiv</span>}
              {(c?.error ?? 0) > 0 && <span className="text-red-600">{c?.error} failed</span>}
              {result.summarizing && <span className="text-indigo-500">· summarizing in background</span>}
            </div>
            <div className="rounded-lg border border-gray-200 divide-y divide-gray-100 max-h-52 overflow-y-auto">
              {result.results.map((r, i) => {
                const m = STATUS_META[r.status];
                const Icon = m.icon;
                return (
                  <div key={i} className="flex items-start gap-2 p-2.5 text-xs">
                    <Icon size={14} className={`mt-0.5 shrink-0 ${m.cls}`} />
                    <span className="min-w-0 flex-1">
                      <span className="text-gray-700 break-all">{r.title || r.arxiv_id || r.query}</span>
                      <span className={`block ${m.cls}`}>{m.label}</span>
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <div className="px-5 py-4 border-t border-gray-100 flex justify-end gap-3">
        <button onClick={onClose} className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900">
          {result ? "Done" : "Cancel"}
        </button>
        <button
          onClick={handleAdd}
          disabled={!text.trim() || loading}
          className="inline-flex items-center gap-2 px-5 py-2.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 font-medium"
        >
          {loading && <Loader2 size={15} className="animate-spin" />}
          {loading ? "Adding…" : result ? "Add more" : "Add papers"}
        </button>
      </div>
    </>
  );
}
