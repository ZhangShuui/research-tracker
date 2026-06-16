"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { X, Sparkles, Loader2 } from "lucide-react";
import { api, Paper, CrossDomainCandidate } from "@/lib/api";

interface Props {
  topicId: string;
  papers: Paper[];
  onClose: () => void;
  /** Called once generation completes (parent clears selection + navigates). */
  onGenerated: (sessionId: string) => void;
}

export function GenerateInsightsModal({ topicId, papers, onClose, onGenerated }: Props) {
  const [candidates, setCandidates] = useState<CrossDomainCandidate[] | null>(null);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [suggesting, setSuggesting] = useState(false);
  const [suggestErr, setSuggestErr] = useState<string | null>(null);

  const [liveSearch, setLiveSearch] = useState(false);
  const [kbCount, setKbCount] = useState<number | null>(null);

  const [generating, setGenerating] = useState(false);
  const [genMode, setGenMode] = useState<"single" | "agentic" | null>(null);
  const [genErr, setGenErr] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (pollRef.current) clearTimeout(pollRef.current); }, []);

  const paperIds = papers.map((p) => p.arxiv_id);
  const thinCount = papers.filter((p) => !p.key_insight && !p.summary).length;

  const handleSuggest = useCallback(async () => {
    if (!papers.length || suggesting) return;
    setSuggesting(true);
    setSuggestErr(null);
    setCandidates(null);
    try {
      const res = await api.suggestCrossDomain(topicId, paperIds, liveSearch);
      setCandidates(res.candidates);
      setKbCount(res.kb_count);
      setChecked(new Set()); // semi-auto: tick which to include
    } catch (e) {
      setSuggestErr(e instanceof Error ? e.message : "Suggestion failed.");
    } finally {
      setSuggesting(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topicId, papers, suggesting, liveSearch]);

  const toggleCandidate = (id: string) =>
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const pollSession = useCallback((sid: string) => {
    const tick = async () => {
      try {
        const s = await api.getSession(topicId, sid);
        if (s.status === "running") {
          pollRef.current = setTimeout(tick, 2500);
        } else if (s.status === "completed") {
          setGenerating(false);
          setGenMode(null);
          onGenerated(sid);
        } else {
          setGenerating(false);
          setGenMode(null);
          setGenErr(s.error_message || "Generation failed.");
        }
      } catch {
        pollRef.current = setTimeout(tick, 2500);
      }
    };
    tick();
  }, [topicId, onGenerated]);

  const handleGenerate = useCallback(async (mode: "single" | "agentic") => {
    if (!papers.length || generating) return;
    setGenerating(true);
    setGenMode(mode);
    setGenErr(null);
    try {
      const picked = (candidates ?? []).filter((c) => checked.has(c.arxiv_id));
      const res = await api.generateInsights(topicId, {
        paper_ids: paperIds,
        cross_domain_papers: picked,
        mode,
      });
      pollSession(res.session_id);
    } catch (e) {
      setGenerating(false);
      setGenMode(null);
      setGenErr(e instanceof Error ? e.message : "Failed to start generation.");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topicId, papers, candidates, checked, generating, pollSession]);

  return (
    <div className="fixed inset-0 bg-black/40 flex items-end sm:items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <h3 className="text-base font-semibold text-gray-900 flex items-center gap-2">
            <Sparkles size={17} className="text-indigo-500" />
            Generate insights
          </h3>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors text-gray-500"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="overflow-y-auto flex-1 p-5 space-y-4">
          {/* Selected papers */}
          <div>
            <p className="text-sm text-gray-700">
              <span className="font-semibold">{papers.length}</span> paper{papers.length === 1 ? "" : "s"} selected
            </p>
            {thinCount > 0 && (
              <p className="text-xs text-amber-600 mt-0.5">
                {thinCount} lack summaries — insights may be thin (run Re-filter or Auto-summarize first for best results).
              </p>
            )}
            <div className="mt-2 flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
              {papers.map((p) => (
                <span
                  key={p.arxiv_id}
                  className="inline-block max-w-[260px] truncate rounded-full bg-gray-100 text-gray-600 text-xs px-2.5 py-1"
                  title={p.title}
                >
                  {p.title}
                </span>
              ))}
            </div>
          </div>

          {/* Cross-domain */}
          <div className="border-t border-gray-100 pt-4">
            <p className="text-xs font-medium text-gray-500 mb-1.5">Cross-domain papers (optional)</p>
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={handleSuggest}
                disabled={suggesting}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50 text-gray-700"
              >
                {suggesting ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                {suggesting ? "Finding…" : "Suggest from knowledge base"}
              </button>
              <label className="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer">
                <input
                  type="checkbox"
                  checked={liveSearch}
                  onChange={(e) => setLiveSearch(e.target.checked)}
                  className="rounded border-gray-300"
                />
                also search arXiv live (keyword)
              </label>
            </div>
            {suggestErr && <p className="mt-1 text-xs text-red-600">{suggestErr}</p>}

            {candidates && candidates.length > 0 && (
              <p className="mt-2 text-[11px] text-gray-400">
                {kbCount ?? 0} from knowledge base
                {candidates.filter((c) => c.random).length > 0
                  ? ` · ${candidates.filter((c) => c.random).length} random 🎲`
                  : ""}
                {liveSearch ? ` · ${Math.max(0, candidates.length - (kbCount ?? 0))} live` : ""}
              </p>
            )}
            {candidates && candidates.length === 0 && (
              <p className="mt-1.5 text-xs text-amber-600">
                {liveSearch
                  ? "No suggestions found."
                  : "Knowledge base returned nothing. Add papers in the Cross-domain Library, or tick “also search arXiv live”."}
              </p>
            )}
            {candidates && candidates.length > 0 && (
              <div className="mt-2 space-y-1.5 max-h-56 overflow-y-auto">
                {candidates.map((c) => (
                  <label
                    key={c.arxiv_id}
                    className="flex items-start gap-2 rounded-lg border border-gray-200 p-2.5 cursor-pointer hover:bg-gray-50"
                  >
                    <input
                      type="checkbox"
                      checked={checked.has(c.arxiv_id)}
                      onChange={() => toggleCandidate(c.arxiv_id)}
                      className="mt-0.5 rounded border-gray-300"
                    />
                    <span className="text-xs text-gray-700 leading-snug">
                      <span className={`mr-1 rounded px-1 text-[9px] font-medium ${
                        c.source === "live" ? "bg-amber-100 text-amber-700" : "bg-indigo-100 text-indigo-700"
                      }`}>
                        {c.source === "live" ? "live" : "KB"}
                      </span>
                      {c.random && (
                        <span
                          className="mr-1 rounded px-1 text-[9px] font-medium bg-purple-100 text-purple-700"
                          title="Random pick for serendipity — a math paper mixed in beyond the closest matches to spark non-obvious connections."
                        >
                          🎲 wildcard
                        </span>
                      )}
                      {c.domain && (
                        <span className="font-mono text-[10px] text-indigo-500 mr-1">{c.domain}</span>
                      )}
                      {c.title}
                      {c.rationale && (
                        <span className="block text-[11px] text-gray-400 mt-0.5">{c.rationale}</span>
                      )}
                    </span>
                  </label>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer — two entry points: single-pass vs deep multi-stage */}
        <div className="px-5 py-4 border-t border-gray-100 space-y-2">
          {genErr && <p className="text-sm text-red-600">{genErr}</p>}
          <p className="text-[11px] text-gray-400 leading-snug">
            <span className="font-medium text-gray-500">Single-pass</span> — one opus call, best for ≤~25 papers (sharpest cross-paper connections).{" "}
            <span className="font-medium text-gray-500">Deep</span> — outline → parallel themes → cross-theme connections → audit; best for large sets.
          </p>
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-gray-400">
              {generating
                ? genMode === "agentic"
                  ? `Running multi-stage pipeline over ${papers.length + checked.size} papers — may take a few minutes.`
                  : `Running one opus call over ${papers.length + checked.size} papers — ~1 min.`
                : papers.length + checked.size > 25
                ? `${papers.length + checked.size} papers — Deep recommended.`
                : `${papers.length + checked.size} paper${papers.length + checked.size === 1 ? "" : "s"} — Single-pass is plenty.`}
            </span>
            <div className="flex items-center gap-2">
              <button onClick={onClose} className="px-3 py-2 text-sm text-gray-600 hover:text-gray-900">
                Cancel
              </button>
              <button
                onClick={() => handleGenerate("single")}
                disabled={!papers.length || generating}
                title="One opus call over all selected papers. Best for ≤~25 papers — sharpest cross-paper connections."
                className="inline-flex items-center gap-1.5 px-4 py-2.5 text-sm rounded-lg border border-indigo-300 text-indigo-700 hover:bg-indigo-50 transition-colors disabled:opacity-50 font-medium"
              >
                {generating && genMode === "single" && <Loader2 size={15} className="animate-spin" />}
                Single-pass
              </button>
              <button
                onClick={() => handleGenerate("agentic")}
                disabled={!papers.length || generating}
                title="Outline → parallel theme synthesis → cross-theme connections → audit + grounding. Best for large selections; scales to 200+ papers."
                className="inline-flex items-center gap-2 px-5 py-2.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 font-medium"
              >
                {generating && genMode === "agentic" ? <Loader2 size={15} className="animate-spin" /> : <Sparkles size={15} />}
                Deep
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
