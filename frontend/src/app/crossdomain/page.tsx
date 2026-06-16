"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft, Sigma, Loader2, Trash2, Wand2, Check, AlertCircle,
  MinusCircle, Search, Download,
} from "lucide-react";
import { api, CorpusAddResponse, CorpusAddResultItem, ImportCandidate } from "@/lib/api";
import { useDebounce } from "@/lib/hooks";
import { Pagination } from "@/components/Pagination";

const PAGE = 30;

const STATUS_META: Record<CorpusAddResultItem["status"], { label: string; cls: string; icon: typeof Check }> = {
  added: { label: "Added", cls: "text-emerald-600", icon: Check },
  skipped: { label: "Skipped (not theoretical)", cls: "text-amber-600", icon: MinusCircle },
  duplicate: { label: "Already in corpus", cls: "text-gray-400", icon: MinusCircle },
  not_found: { label: "Not found on arXiv", cls: "text-amber-600", icon: AlertCircle },
  invalid: { label: "Not an arXiv link/ID", cls: "text-amber-600", icon: AlertCircle },
  error: { label: "Fetch failed", cls: "text-red-600", icon: AlertCircle },
};

export default function CrossDomainPage() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: ["corpus"] });

  // ---- corpus list ----
  const [search, setSearch] = useState("");
  const debounced = useDebounce(search, 300);
  const [offset, setOffset] = useState(0);
  const { data: list } = useQuery({
    queryKey: ["corpus", debounced, offset],
    queryFn: () => api.getCorpusPapers({ search: debounced, limit: PAGE, offset }),
    refetchInterval: 5000, // pick up embedding progress
  });
  const embJob = list?.embedding_job;
  const embedding = embJob?.status === "running";

  // ---- paste add ----
  const [text, setText] = useState("");
  const [skipScreen, setSkipScreen] = useState(false);
  const [addResult, setAddResult] = useState<CorpusAddResponse | null>(null);
  const addMut = useMutation({
    mutationFn: () => api.addCorpusPapers({ text, skip_screen: skipScreen }),
    onSuccess: (res) => { setAddResult(res); invalidate(); },
  });

  // ---- category importer ----
  const [cats, setCats] = useState("");
  const [kw, setKw] = useState("");
  const [maxN, setMaxN] = useState(30);
  const [candidates, setCandidates] = useState<ImportCandidate[] | null>(null);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const importMut = useMutation({
    mutationFn: () => api.importSearchCorpus({
      categories: cats.split(",").map((s) => s.trim()).filter(Boolean),
      keyword: kw.trim(),
      max: maxN,
    }),
    onSuccess: (res) => {
      setCandidates(res.candidates);
      setChecked(new Set(res.candidates.filter((c) => c.keep).map((c) => c.arxiv_id)));
    },
  });
  const addSelectedMut = useMutation({
    mutationFn: () => api.addCorpusPapers({ arxiv_ids: Array.from(checked), skip_screen: true }),
    onSuccess: () => { setCandidates(null); setChecked(new Set()); invalidate(); },
  });

  const delMut = useMutation({
    mutationFn: (id: string) => api.deleteCorpusPaper(id),
    onSuccess: invalidate,
  });
  const buildEmbMut = useMutation({
    mutationFn: () => api.buildCorpusEmbeddings(),
    onSuccess: invalidate,
  });

  // Concept cards (main math content) — generated + re-embedded in the background.
  const { data: cardStatus } = useQuery({
    queryKey: ["corpus-cards"],
    queryFn: () => api.getConceptCardsStatus(),
    refetchInterval: 4_000,
  });
  const cardJob = cardStatus?.job;
  const cardsRunning = cardJob?.status === "running";
  const buildCardsMut = useMutation({
    mutationFn: () => api.buildConceptCards(false),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["corpus-cards"] }),
  });

  const c = addResult?.counts;
  const toggle = (id: string) => setChecked((p) => {
    const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n;
  });

  return (
    <main className="min-h-screen max-w-5xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div>
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-800 mb-2">
          <ArrowLeft size={15} /> Dashboard
        </Link>
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Sigma size={22} className="text-indigo-500" /> Cross-domain Knowledge Base
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          A curated, embedded corpus of authoritative, theory-leaning papers (math / physics /
          stats / foundational CS). Insights generation retrieves from here by vector search.
        </p>
      </div>

      {/* Embedding status bar */}
      <div className="bg-white rounded-xl border border-gray-200 p-4 flex flex-wrap items-center gap-4">
        <div className="text-sm text-gray-700">
          <span className="font-semibold">{list?.total ?? 0}</span> papers ·{" "}
          <span className="font-semibold">{list?.embedding_count ?? 0}</span> embedded
        </div>
        {embJob && embJob.status !== "idle" && (
          <span className={`text-xs ${embedding ? "text-blue-600" : embJob.status === "failed" ? "text-red-600" : "text-emerald-600"}`}>
            {embedding ? "embedding…" : embJob.status === "failed" ? "embedding failed" : `embedded ${embJob.embedded ?? 0}`}
          </span>
        )}
        {cardJob && cardJob.status !== "idle" && (
          <span className={`text-xs ${cardsRunning ? "text-blue-600" : cardJob.status === "failed" ? "text-red-600" : "text-emerald-600"}`}>
            {cardsRunning
              ? `concept cards… ${cardJob.processed ?? 0}/${cardJob.total ?? 0}`
              : cardJob.status === "failed" ? "cards failed" : `cards: ${cardJob.generated ?? 0}`}
          </span>
        )}
        <button
          onClick={() => buildCardsMut.mutate()}
          disabled={cardsRunning || buildCardsMut.isPending}
          className="ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 font-medium"
          title="Extract each paper's main mathematical content (and re-embed)"
        >
          {cardsRunning ? <Loader2 size={13} className="animate-spin" /> : <Sigma size={13} />}
          Generate concept cards
        </button>
        <button
          onClick={() => buildEmbMut.mutate()}
          disabled={embedding || buildEmbMut.isPending}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50 text-gray-700"
        >
          {embedding ? <Loader2 size={13} className="animate-spin" /> : <Wand2 size={13} />}
          Rebuild embeddings
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Paste add */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-3">
          <h2 className="text-sm font-semibold text-gray-900">Add papers</h2>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={"arXiv IDs/URLs or PDF URLs (comma / newline separated)\ne.g. 2401.12345, https://maths.example.edu/paper.pdf"}
            className="input h-28 resize-none font-mono text-xs"
          />
          <p className="text-xs text-gray-400">
            Accepts arXiv IDs/links and direct <span className="font-medium">PDF URLs</span> (for math/theory papers not on arXiv — title/abstract are auto-extracted).
          </p>
          <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
            <input type="checkbox" checked={skipScreen} onChange={(e) => setSkipScreen(e.target.checked)} className="rounded border-gray-300" />
            Skip the theory/authority screen (add exactly what I paste)
          </label>
          <button
            onClick={() => addMut.mutate()}
            disabled={!text.trim() || addMut.isPending}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 font-medium"
          >
            {addMut.isPending && <Loader2 size={14} className="animate-spin" />}
            {addMut.isPending ? "Screening & adding…" : "Add to corpus"}
          </button>
          {addResult && (
            <div className="space-y-2">
              <div className="flex flex-wrap gap-3 text-xs">
                <span className="text-emerald-600 font-medium">{c?.added ?? 0} added</span>
                {(c?.skipped ?? 0) > 0 && <span className="text-amber-600">{c?.skipped} skipped</span>}
                {(c?.duplicate ?? 0) > 0 && <span className="text-gray-400">{c?.duplicate} dup</span>}
                {(c?.not_found ?? 0) > 0 && <span className="text-amber-600">{c?.not_found} not found</span>}
                {(c?.invalid ?? 0) > 0 && <span className="text-amber-600">{c?.invalid} invalid</span>}
                {(c?.error ?? 0) > 0 && <span className="text-red-600">{c?.error} failed</span>}
              </div>
              <div className="rounded-lg border border-gray-200 divide-y divide-gray-100 max-h-40 overflow-y-auto">
                {addResult.results.map((r, i) => {
                  const m = STATUS_META[r.status]; const Icon = m.icon;
                  return (
                    <div key={i} className="flex items-start gap-2 p-2 text-xs">
                      <Icon size={13} className={`mt-0.5 shrink-0 ${m.cls}`} />
                      <span className="min-w-0">
                        <span className="text-gray-700 break-all">{r.title || r.arxiv_id || r.query}</span>
                        <span className={`block ${m.cls}`}>{m.label}{r.reason ? ` — ${r.reason}` : ""}</span>
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Category importer */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-3">
          <h2 className="text-sm font-semibold text-gray-900">Import from arXiv</h2>
          <input
            value={cats}
            onChange={(e) => setCats(e.target.value)}
            placeholder="Categories (comma) — blank = math.OC, stat.ML, math.PR, …"
            className="input text-xs"
          />
          <div className="flex gap-2">
            <input
              value={kw}
              onChange={(e) => setKw(e.target.value)}
              placeholder="Keyword (optional, all-time search)"
              className="input text-xs flex-1"
            />
            <input
              type="number" min={1} max={100} value={maxN}
              onChange={(e) => setMaxN(Number(e.target.value))}
              className="input text-xs w-20"
              title="Max candidates"
            />
          </div>
          <button
            onClick={() => importMut.mutate()}
            disabled={importMut.isPending}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50 text-gray-700"
          >
            {importMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
            {importMut.isPending ? "Searching & screening…" : "Search candidates"}
          </button>

          {candidates && candidates.length === 0 && (
            <p className="text-xs text-gray-400">No new candidates found.</p>
          )}
          {candidates && candidates.length > 0 && (
            <>
              <div className="space-y-1.5 max-h-60 overflow-y-auto">
                {candidates.map((c2) => (
                  <label key={c2.arxiv_id} className={`flex items-start gap-2 rounded-lg border p-2 cursor-pointer ${c2.keep ? "border-emerald-200 bg-emerald-50/40" : "border-gray-200"}`}>
                    <input type="checkbox" checked={checked.has(c2.arxiv_id)} onChange={() => toggle(c2.arxiv_id)} className="mt-0.5 rounded border-gray-300" />
                    <span className="text-xs text-gray-700 leading-snug min-w-0">
                      {c2.domain && <span className="font-mono text-[10px] text-indigo-500 mr-1">{c2.domain}</span>}
                      {!c2.keep && <span className="text-[10px] text-amber-600 mr-1">(screen: skip)</span>}
                      {c2.title}
                      {c2.reason && <span className="block text-[11px] text-gray-400 mt-0.5">{c2.reason}</span>}
                    </span>
                  </label>
                ))}
              </div>
              <button
                onClick={() => addSelectedMut.mutate()}
                disabled={checked.size === 0 || addSelectedMut.isPending}
                className="inline-flex items-center gap-2 px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 font-medium"
              >
                {addSelectedMut.isPending && <Loader2 size={14} className="animate-spin" />}
                <Download size={14} /> Add selected ({checked.size})
              </button>
            </>
          )}
        </div>
      </div>

      {/* Corpus list */}
      <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-gray-900">Corpus</h2>
          <div className="relative w-64 max-w-full">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              value={search}
              onChange={(e) => { setSearch(e.target.value); setOffset(0); }}
              placeholder="Search corpus…"
              className="input pl-9 text-sm"
            />
          </div>
        </div>
        {!list?.papers.length ? (
          <p className="text-sm text-gray-400 py-8 text-center">
            {debounced ? "No matching papers." : "Corpus is empty — add papers above."}
          </p>
        ) : (
          <div className="divide-y divide-gray-100">
            {list.papers.map((p) => (
              <div key={p.arxiv_id} className="flex items-start gap-3 py-2.5">
                <div className="min-w-0 flex-1">
                  <a href={p.url} target="_blank" rel="noopener noreferrer" className="text-sm text-gray-900 hover:text-indigo-600 line-clamp-1 font-medium">
                    {p.title}
                  </a>
                  <div className="text-xs text-gray-400 mt-0.5 flex items-center gap-2 flex-wrap">
                    {p.venue && <span className="font-mono text-indigo-500">{p.venue}</span>}
                    <span className="line-clamp-1">{p.authors}</span>
                  </div>
                  {/* Concept card: math summary + key concepts */}
                  {p.summary && (
                    <p className="text-xs text-gray-600 mt-1 line-clamp-2">{p.summary}</p>
                  )}
                  {p.math_concepts && p.math_concepts.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {p.math_concepts.slice(0, 6).map((mc) => (
                        <span key={mc} className="text-[10px] px-1.5 py-0.5 rounded bg-purple-50 text-purple-700">
                          {mc}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => { if (confirm(`Remove "${p.title.slice(0, 60)}…" from the corpus?`)) delMut.mutate(p.arxiv_id); }}
                  className="p-1.5 text-gray-400 hover:text-red-600 transition-colors shrink-0"
                  title="Remove from corpus"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
        {list && list.total > PAGE && (
          <Pagination total={list.total} limit={PAGE} offset={offset} onChange={setOffset} />
        )}
      </div>
    </main>
  );
}
