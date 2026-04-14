"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  X,
  Send,
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  Loader2,
} from "lucide-react";
import { api, TopicCreate, GuidedEvent, guidedCreateStream } from "@/lib/api";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Props {
  onClose: () => void;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

/* ------------------------------------------------------------------ */
/*  Field helper                                                       */
/* ------------------------------------------------------------------ */

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-sm font-medium text-gray-700">{label}</label>
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  MessageBubble                                                      */
/* ------------------------------------------------------------------ */

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-slate-100 text-gray-900"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  GuidedCreateModal (Phase 1 — Chat)                                 */
/* ------------------------------------------------------------------ */

export function GuidedCreateModal({ onClose }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [draftConfig, setDraftConfig] = useState<TopicCreate | null>(null);
  const [phase, setPhase] = useState<"chat" | "preview">("chat");

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new content
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingText]);

  // Abort on unmount / close
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || streaming) return;

    const userMsg: Message = { role: "user", content: text };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput("");
    setStreaming(true);
    setStreamingText("");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let accumulated = "";
      await guidedCreateStream(
        nextMessages.map((m) => ({ role: m.role, content: m.content })),
        (event: GuidedEvent) => {
          if (event.type === "delta") {
            accumulated += event.data.text;
            setStreamingText(accumulated);
          } else if (event.type === "done") {
            const assistantMsg: Message = {
              role: "assistant",
              content: event.data.message,
            };
            setMessages((prev) => [...prev, assistantMsg]);
            setStreamingText("");
            if (
              event.data.stage === "ready" &&
              event.data.draft_config
            ) {
              setDraftConfig(event.data.draft_config);
              setPhase("preview");
            }
          }
        },
        controller.signal,
      );
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") return;
      // If the stream errored, surface it as an assistant message
      const errMsg =
        err instanceof Error ? err.message : "Something went wrong";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errMsg}` },
      ]);
      setStreamingText("");
    } finally {
      setStreaming(false);
    }
  }, [input, streaming, messages]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  // ----- Phase 2: Preview -----
  if (phase === "preview" && draftConfig) {
    return (
      <GuidedPreview
        draftConfig={draftConfig}
        onBack={() => setPhase("chat")}
        onClose={onClose}
      />
    );
  }

  // ----- Phase 1: Chat -----
  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b shrink-0">
          <h2 className="font-semibold text-lg">Guided Topic Creation</h2>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Messages area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 space-y-4">
          {messages.length === 0 && !streaming && (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-400 text-sm">
                Tell me about the research area you want to track...
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}

          {/* Streaming assistant text */}
          {streaming && streamingText && (
            <div className="flex justify-start">
              <div className="max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap bg-slate-100 text-gray-900">
                {streamingText}
              </div>
            </div>
          )}

          {/* Thinking loader */}
          {streaming && !streamingText && (
            <div className="flex justify-start">
              <div className="flex items-center gap-2 rounded-2xl px-4 py-2.5 bg-slate-100 text-gray-500 text-sm">
                <Loader2 size={14} className="animate-spin" />
                Thinking...
              </div>
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="border-t p-4 shrink-0">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe your research interest..."
              rows={2}
              className="input flex-1 resize-none text-sm"
              disabled={streaming}
            />
            <button
              onClick={sendMessage}
              disabled={streaming || !input.trim()}
              className="p-2.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 shrink-0"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  GuidedPreview (Phase 2 — Editable Preview)                         */
/* ------------------------------------------------------------------ */

function GuidedPreview({
  draftConfig,
  onBack,
  onClose,
}: {
  draftConfig: TopicCreate;
  onBack: () => void;
  onClose: () => void;
}) {
  const qc = useQueryClient();

  // Form state initialized from draftConfig
  const [name, setName] = useState(draftConfig.name ?? "");
  const [description, setDescription] = useState(
    draftConfig.description ?? "",
  );
  const [arxivKeywords, setArxivKeywords] = useState(
    (draftConfig.arxiv_keywords ?? []).join("\n"),
  );
  const [arxivCategories, setArxivCategories] = useState(
    (draftConfig.arxiv_categories ?? []).join(", "),
  );
  const [arxivLookback, setArxivLookback] = useState(
    draftConfig.arxiv_lookback_days ?? 2,
  );
  const [githubKeywords, setGithubKeywords] = useState(
    (draftConfig.github_keywords ?? []).join("\n"),
  );
  const [githubLookback, setGithubLookback] = useState(
    draftConfig.github_lookback_days ?? 7,
  );
  const [scheduleCron, setScheduleCron] = useState(
    draftConfig.schedule_cron ?? "",
  );

  // Date range
  const [searchDateFrom, setSearchDateFrom] = useState(
    draftConfig.search_date_from ?? "",
  );
  const [searchDateTo, setSearchDateTo] = useState(
    draftConfig.search_date_to ?? "",
  );

  // Additional sources
  const [showSources, setShowSources] = useState(
    Boolean(draftConfig.openalex_enabled || draftConfig.openreview_enabled),
  );

  // OpenAlex
  const [oaEnabled, setOaEnabled] = useState(
    draftConfig.openalex_enabled ?? false,
  );
  const [oaKeywords, setOaKeywords] = useState(
    (draftConfig.openalex_keywords ?? []).join("\n"),
  );
  const [oaLookback, setOaLookback] = useState(
    draftConfig.openalex_lookback_days ?? 7,
  );
  const [oaVenues, setOaVenues] = useState(
    (draftConfig.openalex_venues ?? []).join(", "),
  );
  const [oaMaxResults, setOaMaxResults] = useState(
    draftConfig.openalex_max_results ?? 200,
  );

  // OpenReview
  const [orEnabled, setOrEnabled] = useState(
    draftConfig.openreview_enabled ?? false,
  );
  const [orVenues, setOrVenues] = useState(
    (draftConfig.openreview_venues ?? []).join(", "),
  );
  const [orKeywords, setOrKeywords] = useState(
    (draftConfig.openreview_keywords ?? []).join("\n"),
  );
  const [orMaxResults, setOrMaxResults] = useState(
    draftConfig.openreview_max_results ?? 100,
  );

  const [error, setError] = useState("");

  const createMut = useMutation({
    mutationFn: (body: TopicCreate) => api.createTopic(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["topics"] });
      onClose();
    },
    onError: (e: Error) => setError(e.message),
  });

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    const body: TopicCreate = {
      name: name.trim(),
      description: description.trim(),
      arxiv_keywords: arxivKeywords
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      arxiv_categories: arxivCategories
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      arxiv_lookback_days: Number(arxivLookback),
      github_keywords: githubKeywords
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      github_lookback_days: Number(githubLookback),
      schedule_cron: scheduleCron.trim(),
      enabled: true,
      // OpenAlex
      openalex_enabled: oaEnabled,
      openalex_keywords: oaKeywords
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      openalex_lookback_days: Number(oaLookback),
      openalex_venues: oaVenues
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      openalex_max_results: Number(oaMaxResults),
      // OpenReview
      openreview_enabled: orEnabled,
      openreview_venues: orVenues
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      openreview_keywords: orKeywords
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      openreview_max_results: Number(orMaxResults),
      // Date range
      search_date_from: searchDateFrom,
      search_date_to: searchDateTo,
    };
    createMut.mutate(body);
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b sticky top-0 bg-white rounded-t-2xl z-10">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <ArrowLeft size={18} />
            </button>
            <h2 className="font-semibold text-lg">Review &amp; Create</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <Field label="Name *">
            <input
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input"
              placeholder="e.g. Video Diffusion Models"
            />
          </Field>
          <Field label="Description">
            <input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="input"
              placeholder="Brief description"
            />
          </Field>
          <Field label="arXiv Keywords (one per line)">
            <textarea
              value={arxivKeywords}
              onChange={(e) => setArxivKeywords(e.target.value)}
              className="input h-24 resize-none"
              placeholder={"video diffusion\nworld model\n..."}
            />
          </Field>
          <Field label="arXiv Categories (comma-separated)">
            <input
              value={arxivCategories}
              onChange={(e) => setArxivCategories(e.target.value)}
              className="input"
              placeholder="cs.CV, cs.AI, cs.LG"
            />
          </Field>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Date From (overrides lookback)">
              <input
                type="date"
                value={searchDateFrom}
                onChange={(e) => setSearchDateFrom(e.target.value)}
                className="input"
              />
            </Field>
            <Field label="Date To">
              <input
                type="date"
                value={searchDateTo}
                onChange={(e) => setSearchDateTo(e.target.value)}
                className="input"
              />
            </Field>
          </div>
          <p className="text-xs text-gray-400 -mt-3">
            Leave empty to use lookback days. When set, date range applies to
            all sources.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <Field label="arXiv Lookback Days">
              <input
                type="number"
                min={1}
                max={365}
                value={arxivLookback}
                onChange={(e) => setArxivLookback(Number(e.target.value))}
                className="input"
              />
            </Field>
            <Field label="GitHub Lookback Days">
              <input
                type="number"
                min={1}
                max={365}
                value={githubLookback}
                onChange={(e) => setGithubLookback(Number(e.target.value))}
                className="input"
              />
            </Field>
          </div>
          <Field label="GitHub Keywords (one per line)">
            <textarea
              value={githubKeywords}
              onChange={(e) => setGithubKeywords(e.target.value)}
              className="input h-20 resize-none"
              placeholder={"video diffusion model\nworld model\n..."}
            />
          </Field>
          <Field label="Schedule (cron, optional)">
            <input
              value={scheduleCron}
              onChange={(e) => setScheduleCron(e.target.value)}
              className="input font-mono"
              placeholder="0 8 * * * (daily at 8am)"
            />
            <p className="text-xs text-gray-400 mt-1">
              Leave empty for manual-only. Format: min hour day month weekday
            </p>
          </Field>

          {/* Additional Sources */}
          <div className="border rounded-xl overflow-hidden">
            <button
              type="button"
              onClick={() => setShowSources(!showSources)}
              className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            >
              <span>Additional Sources</span>
              {showSources ? (
                <ChevronDown size={16} />
              ) : (
                <ChevronRight size={16} />
              )}
            </button>
            {showSources && (
              <div className="px-4 pb-4 space-y-4 border-t">
                {/* OpenAlex */}
                <div className="pt-3">
                  <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
                    <input
                      type="checkbox"
                      checked={oaEnabled}
                      onChange={(e) => setOaEnabled(e.target.checked)}
                      className="rounded"
                    />
                    OpenAlex
                  </label>
                  {oaEnabled && (
                    <div className="ml-6 space-y-3">
                      <Field label="Keywords (one per line, empty = use arXiv keywords)">
                        <textarea
                          value={oaKeywords}
                          onChange={(e) => setOaKeywords(e.target.value)}
                          className="input h-16 resize-none text-sm"
                          placeholder="(leave empty to reuse arXiv keywords)"
                        />
                      </Field>
                      <Field label="Venue Filter (comma-separated)">
                        <input
                          value={oaVenues}
                          onChange={(e) => setOaVenues(e.target.value)}
                          className="input text-sm"
                          placeholder="Pattern Analysis, Computing Surveys, NeurIPS"
                        />
                        <p className="text-xs text-gray-400 mt-1">
                          Matches venue name substrings. Use OpenAlex source
                          names (e.g. &quot;Pattern Analysis&quot; for TPAMI).
                        </p>
                      </Field>
                      <div className="grid grid-cols-2 gap-3">
                        <Field label="Lookback Days">
                          <input
                            type="number"
                            min={1}
                            max={365}
                            value={oaLookback}
                            onChange={(e) =>
                              setOaLookback(Number(e.target.value))
                            }
                            className="input text-sm"
                          />
                        </Field>
                        <Field label="Max Results">
                          <input
                            type="number"
                            min={10}
                            max={1000}
                            value={oaMaxResults}
                            onChange={(e) =>
                              setOaMaxResults(Number(e.target.value))
                            }
                            className="input text-sm"
                          />
                        </Field>
                      </div>
                    </div>
                  )}
                </div>

                {/* OpenReview */}
                <div className="pt-1 border-t">
                  <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2 mt-3">
                    <input
                      type="checkbox"
                      checked={orEnabled}
                      onChange={(e) => setOrEnabled(e.target.checked)}
                      className="rounded"
                    />
                    OpenReview
                  </label>
                  {orEnabled && (
                    <div className="ml-6 space-y-3">
                      <Field label="Venues (comma-separated short names)">
                        <input
                          value={orVenues}
                          onChange={(e) => setOrVenues(e.target.value)}
                          className="input text-sm"
                          placeholder="iclr2025, neurips2024, icml2025"
                        />
                        <p className="text-xs text-gray-400 mt-1">
                          Known: iclr2024/2025, neurips2024/2025, icml2024/2025,
                          acl2024/2025
                        </p>
                      </Field>
                      <Field label="Keywords (one per line, empty = use arXiv keywords)">
                        <textarea
                          value={orKeywords}
                          onChange={(e) => setOrKeywords(e.target.value)}
                          className="input h-16 resize-none text-sm"
                          placeholder="(leave empty to reuse arXiv keywords)"
                        />
                      </Field>
                      <Field label="Max Results per Venue">
                        <input
                          type="number"
                          min={10}
                          max={500}
                          value={orMaxResults}
                          onChange={(e) =>
                            setOrMaxResults(Number(e.target.value))
                          }
                          className="input text-sm"
                        />
                      </Field>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {error && (
            <p className="text-sm text-red-600 bg-red-50 rounded-lg p-3">
              {error}
            </p>
          )}

          <div className="flex justify-between pt-2">
            <button
              type="button"
              onClick={onBack}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              Back to Chat
            </button>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={createMut.isPending}
                className="px-5 py-2.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 font-medium"
              >
                {createMut.isPending ? "Creating..." : "Create Topic"}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
