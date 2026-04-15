"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useSearchParams } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BookOpen,
  Search,
  Send,
  Trash2,
  ChevronDown,
  ChevronUp,
  Loader2,
  Check,
  Circle,
  Globe,
} from "lucide-react";
import { api, Paper, DeepReadSession, DeepReadSessionDetail, DeepReadMessage } from "@/lib/api";
import { useDebounce } from "@/lib/hooks";
import { MathMarkdown } from "@/components/MathMarkdown";

const ANALYSIS_SECTIONS = [
  { key: "motivation_analysis", label: "Motivation" },
  { key: "method_analysis", label: "Method" },
  { key: "experiment_analysis", label: "Experiments" },
  { key: "contribution_analysis", label: "Contribution" },
  { key: "code_review", label: "Code Review" },
] as const;

const PROGRESS_STEPS = [
  { key: "motivation_analysis", label: "Motivation" },
  { key: "method_analysis", label: "Method" },
  { key: "experiment_analysis", label: "Experiments" },
  { key: "contribution_analysis", label: "Contribution" },
  { key: "summary_card_json", label: "Summary Card" },
  { key: "code_review", label: "Code Review" },
] as const;

export default function DeepReadPage() {
  const { id: topicId } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();

  // Paper selection
  const [paperSearch, setPaperSearch] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(
    searchParams.get("paper") || null
  );
  const [selectedPaperTitle, setSelectedPaperTitle] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Session selection
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // Analysis tab
  const [activeTab, setActiveTab] = useState<string>("motivation_analysis");

  // Notes
  const [notes, setNotes] = useState("");
  const [notesSaved, setNotesSaved] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);
  const notesSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // QA
  const [qaInput, setQaInput] = useState("");
  const [pendingMsgId, setPendingMsgId] = useState<string | null>(null);

  // Starting deep read
  const [isStarting, setIsStarting] = useState(false);
  const [language, setLanguage] = useState<"en" | "zh">("en");

  const debouncedPaperSearch = useDebounce(paperSearch, 300);

  // --- Paper search dropdown ---
  const { data: paperResults } = useQuery({
    queryKey: ["papers-search", topicId, debouncedPaperSearch],
    queryFn: () => api.getPapers(topicId, { search: debouncedPaperSearch, limit: 20 }),
    enabled: showDropdown && debouncedPaperSearch.length > 0,
  });

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  // --- Session list ---
  const { data: sessionsData } = useQuery({
    queryKey: ["deep-read-sessions", topicId, selectedPaperId],
    queryFn: () => api.listDeepReadSessions(topicId, selectedPaperId || undefined),
    refetchInterval: 5000,
  });
  const sessions = sessionsData?.sessions ?? [];

  // --- Active session detail ---
  const { data: sessionDetail } = useQuery({
    queryKey: ["deep-read-session", topicId, activeSessionId],
    queryFn: () => api.getDeepReadSession(topicId, activeSessionId!),
    enabled: !!activeSessionId,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === "running" ? 3000 : false;
    },
  });

  // --- Progress polling ---
  const { data: progressData } = useQuery({
    queryKey: ["deep-read-progress", topicId, activeSessionId],
    queryFn: () => api.getDeepReadProgress(topicId, activeSessionId!),
    enabled: !!activeSessionId && sessionDetail?.status === "running",
    refetchInterval: 2000,
  });

  // --- QA message progress polling ---
  useQuery({
    queryKey: ["deep-read-qa-progress", topicId, activeSessionId, pendingMsgId],
    queryFn: async () => {
      const res = await api.getDeepReadMessageProgress(topicId, activeSessionId!, pendingMsgId!);
      if (res.status === "completed" || res.status === "failed") {
        setPendingMsgId(null);
        queryClient.invalidateQueries({ queryKey: ["deep-read-session", topicId, activeSessionId] });
      }
      return res;
    },
    enabled: !!pendingMsgId && !!activeSessionId,
    refetchInterval: 2000,
  });

  // Sync notes from session detail
  useEffect(() => {
    if (sessionDetail) {
      setNotes(sessionDetail.user_notes || "");
    }
  }, [sessionDetail?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-select paper title if we came via URL param
  useEffect(() => {
    if (selectedPaperId && !selectedPaperTitle) {
      api.getPaper(topicId, selectedPaperId).then((p) => {
        setSelectedPaperTitle(p.title);
      }).catch(() => {});
    }
  }, [selectedPaperId, selectedPaperTitle, topicId]);

  // --- Handlers ---
  function handleSelectPaper(paper: Paper) {
    setSelectedPaperId(paper.arxiv_id);
    setSelectedPaperTitle(paper.title);
    setPaperSearch("");
    setShowDropdown(false);
    setActiveSessionId(null);
  }

  async function handleStartDeepRead() {
    if (!selectedPaperId || isStarting) return;
    setIsStarting(true);
    try {
      const res = await api.startDeepRead(topicId, selectedPaperId, language);
      setActiveSessionId(res.session_id);
      queryClient.invalidateQueries({ queryKey: ["deep-read-sessions", topicId] });
    } catch (e) {
      console.error("Failed to start deep read:", e);
    } finally {
      setIsStarting(false);
    }
  }

  async function handleDeleteSession(sid: string) {
    if (!confirm("Delete this deep read session?")) return;
    try {
      await api.deleteDeepReadSession(topicId, sid);
      if (activeSessionId === sid) setActiveSessionId(null);
      queryClient.invalidateQueries({ queryKey: ["deep-read-sessions", topicId] });
    } catch (e) {
      console.error("Failed to delete session:", e);
    }
  }

  const handleNotesChange = useCallback((value: string) => {
    setNotes(value);
    setNotesSaved(false);
    if (notesSaveTimer.current) clearTimeout(notesSaveTimer.current);
    notesSaveTimer.current = setTimeout(async () => {
      if (activeSessionId) {
        try {
          await api.updateDeepReadNotes(topicId, activeSessionId, value);
          setNotesSaved(true);
          setTimeout(() => setNotesSaved(false), 2000);
        } catch (e) {
          console.error("Failed to save notes:", e);
        }
      }
    }, 1000);
  }, [activeSessionId, topicId]);

  async function handleSendQA() {
    if (!qaInput.trim() || !activeSessionId) return;
    const content = qaInput.trim();
    setQaInput("");
    try {
      const res = await api.sendDeepReadMessage(topicId, activeSessionId, content);
      setPendingMsgId(res.assistant_msg_id);
      queryClient.invalidateQueries({ queryKey: ["deep-read-session", topicId, activeSessionId] });
    } catch (e) {
      console.error("Failed to send message:", e);
    }
  }

  const isCompleted = sessionDetail?.status === "completed";
  const isRunning = sessionDetail?.status === "running";
  const sectionsDone = progressData?.sections_done ?? [];

  const summaryCard = sessionDetail?.summary_card_json;
  const hasSummaryCard = summaryCard && Object.keys(summaryCard).length > 0 &&
    (summaryCard.key_innovations?.length > 0 || summaryCard.strengths?.length > 0);

  return (
    <div className="flex gap-4 min-h-[600px]">
      {/* Left sidebar — session list */}
      <div className="w-64 flex-shrink-0 space-y-3">
        {/* Paper selector */}
        <div className="relative" ref={dropdownRef}>
          <div className="flex items-center gap-2">
            <div className="flex-1 relative">
              <Search size={14} className="absolute left-2.5 top-2.5 text-gray-400" />
              <input
                type="text"
                value={paperSearch}
                onChange={(e) => {
                  setPaperSearch(e.target.value);
                  setShowDropdown(true);
                }}
                onFocus={() => paperSearch && setShowDropdown(true)}
                placeholder="Search papers..."
                className="w-full pl-8 pr-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
          {showDropdown && paperResults?.papers?.length ? (
            <div className="absolute z-20 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
              {paperResults.papers.map((p) => (
                <button
                  key={p.arxiv_id}
                  onClick={() => handleSelectPaper(p)}
                  className="w-full text-left px-3 py-2 text-xs hover:bg-blue-50 transition-colors border-b border-gray-100 last:border-0"
                >
                  <div className="font-medium text-gray-900 line-clamp-2">{p.title}</div>
                  <div className="text-gray-400 mt-0.5">{p.arxiv_id}</div>
                </button>
              ))}
            </div>
          ) : null}
        </div>

        {selectedPaperTitle && (
          <div className="bg-blue-50 rounded-lg px-3 py-2 text-xs text-blue-800">
            <span className="font-medium">Selected:</span>{" "}
            <span className="line-clamp-2">{selectedPaperTitle}</span>
          </div>
        )}

        {/* Language selector */}
        <div className="flex items-center gap-1.5 bg-gray-100 rounded-lg p-0.5">
          <button
            onClick={() => setLanguage("en")}
            className={`flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs font-medium rounded-md transition-colors ${
              language === "en"
                ? "bg-white text-gray-900 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            EN
          </button>
          <button
            onClick={() => setLanguage("zh")}
            className={`flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs font-medium rounded-md transition-colors ${
              language === "zh"
                ? "bg-white text-gray-900 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            中文
          </button>
        </div>

        <button
          onClick={handleStartDeepRead}
          disabled={!selectedPaperId || isStarting}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isStarting ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <BookOpen size={14} />
          )}
          Start Deep Read
        </button>

        {/* Session list */}
        <div className="space-y-1">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide px-1">
            Sessions
          </h3>
          {sessions.length === 0 ? (
            <p className="text-xs text-gray-400 px-1">No sessions yet</p>
          ) : (
            sessions.map((s) => (
              <div
                key={s.id}
                onClick={() => setActiveSessionId(s.id)}
                className={`group flex items-start justify-between rounded-lg px-3 py-2 cursor-pointer transition-colors ${
                  activeSessionId === s.id
                    ? "bg-blue-50 border border-blue-200"
                    : "hover:bg-gray-50 border border-transparent"
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium text-gray-900 line-clamp-1">
                    {s.paper_title || s.paper_id}
                  </div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-[10px] text-gray-400 font-mono">{s.id}</span>
                    {s.language === "zh" && (
                      <span className="text-[9px] px-1 py-0.5 bg-orange-100 text-orange-700 rounded font-medium">中</span>
                    )}
                    {s.repo_urls?.length > 0 && (
                      <span className="text-[9px] px-1 py-0.5 bg-violet-100 text-violet-700 rounded font-medium">{s.repo_urls.length} repo{s.repo_urls.length > 1 ? "s" : ""}</span>
                    )}
                    <span
                      className={`inline-block w-1.5 h-1.5 rounded-full ${
                        s.status === "completed"
                          ? "bg-emerald-500"
                          : s.status === "running"
                          ? "bg-amber-500 animate-pulse"
                          : "bg-red-500"
                      }`}
                    />
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteSession(s.id);
                  }}
                  className="p-0.5 text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 min-w-0 space-y-4">
        {!activeSessionId ? (
          <div className="flex items-center justify-center h-96 text-gray-400 text-sm">
            Select or start a deep reading session
          </div>
        ) : !sessionDetail ? (
          <div className="flex items-center justify-center h-96">
            <Loader2 className="animate-spin text-gray-400" />
          </div>
        ) : (
          <>
            {/* Progress indicator */}
            {(isRunning || isCompleted) && (
              <div className="flex items-center gap-1.5 px-1">
                {PROGRESS_STEPS.map((step, i) => {
                  const done = isCompleted || sectionsDone.includes(step.key);
                  const current = progressData?.current_section === step.key;
                  return (
                    <div key={step.key} className="flex items-center gap-1.5">
                      <div className="flex items-center gap-1">
                        {done ? (
                          <Check size={12} className="text-emerald-500" />
                        ) : current ? (
                          <Loader2 size={12} className="text-amber-500 animate-spin" />
                        ) : (
                          <Circle size={12} className="text-gray-300" />
                        )}
                        <span
                          className={`text-[11px] ${
                            done
                              ? "text-emerald-700 font-medium"
                              : current
                              ? "text-amber-700 font-medium"
                              : "text-gray-400"
                          }`}
                        >
                          {step.label}
                        </span>
                      </div>
                      {i < PROGRESS_STEPS.length - 1 && (
                        <div
                          className={`w-4 h-px ${done ? "bg-emerald-300" : "bg-gray-200"}`}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* Summary Card */}
            {isCompleted && hasSummaryCard && (
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-3">Summary Card</h3>
                <div className="grid grid-cols-2 gap-4">
                  <CardList
                    title="Key Innovations"
                    items={summaryCard.key_innovations}
                    color="blue"
                  />
                  <CardList
                    title="Strengths"
                    items={summaryCard.strengths}
                    color="emerald"
                  />
                  <CardList
                    title="Weaknesses"
                    items={summaryCard.weaknesses}
                    color="amber"
                  />
                  <CardList
                    title="Field Comparison"
                    items={summaryCard.field_comparison}
                    color="purple"
                  />
                  <div className="col-span-2">
                    <CardList
                      title="Inspiration Points"
                      items={summaryCard.inspiration_points}
                      color="rose"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Analysis Tabs */}
            <div className="bg-white rounded-xl border border-gray-200">
              <div className="flex border-b border-gray-200">
                {ANALYSIS_SECTIONS.map((section) => {
                  const done = isCompleted || sectionsDone.includes(section.key);
                  const current = progressData?.current_section === section.key;
                  return (
                    <button
                      key={section.key}
                      onClick={() => setActiveTab(section.key)}
                      className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                        activeTab === section.key
                          ? "border-blue-600 text-blue-600"
                          : "border-transparent text-gray-500 hover:text-gray-700"
                      }`}
                    >
                      {section.label}
                      {done && <Check size={12} className="text-emerald-500" />}
                      {current && <Loader2 size={12} className="text-amber-500 animate-spin" />}
                    </button>
                  );
                })}
              </div>
              <div className="p-5 prose prose-sm max-w-none min-h-[200px]">
                {/* Repo info banner for Code Review tab */}
                {activeTab === "code_review" && sessionDetail?.repo_urls?.length > 0 && (
                  <div className="mb-4 space-y-2 not-prose">
                    <div className="flex flex-wrap items-center gap-2 bg-gray-50 rounded-lg px-3 py-2 text-xs">
                      <span className="font-medium text-gray-600">Repos:</span>
                      {sessionDetail.repo_urls.map((url: string) => (
                        <a
                          key={url}
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 px-2 py-0.5 bg-white border border-gray-200 rounded text-blue-600 hover:bg-blue-50 transition-colors"
                        >
                          {url.replace("https://github.com/", "")}
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" /></svg>
                        </a>
                      ))}
                    </div>
                    {sessionDetail.repo_local_paths?.length > 0 && (
                      <div className="flex flex-wrap items-center gap-2 bg-violet-50 rounded-lg px-3 py-2 text-xs">
                        <span className="font-medium text-violet-700">Local clones:</span>
                        {sessionDetail.repo_local_paths.map((p: string) => (
                          <code key={p} className="px-1.5 py-0.5 bg-white border border-violet-200 rounded text-violet-700 font-mono text-[11px]">
                            {p}
                          </code>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                {activeTab === "code_review" && (!sessionDetail?.repo_urls || sessionDetail.repo_urls.length === 0) && isCompleted && (
                  <div className="mb-4 bg-amber-50 rounded-lg px-3 py-2 text-xs text-amber-700 not-prose">
                    No GitHub repositories detected in this paper.
                  </div>
                )}
                {(() => {
                  const content = sessionDetail[activeTab as keyof typeof sessionDetail] as string;
                  if (!content) {
                    const current = progressData?.current_section === activeTab;
                    return (
                      <div className="flex items-center justify-center h-32 text-gray-400 text-sm">
                        {current ? (
                          <span className="flex items-center gap-2">
                            <Loader2 size={16} className="animate-spin" />
                            Analysing...
                          </span>
                        ) : isRunning ? (
                          "Waiting..."
                        ) : (
                          "No analysis available"
                        )}
                      </div>
                    );
                  }
                  return (
                    <MathMarkdown className="prose prose-sm max-w-none">
                      {content}
                    </MathMarkdown>
                  );
                })()}
              </div>
            </div>

            {/* Notes (collapsible) */}
            <div className="bg-white rounded-xl border border-gray-200">
              <button
                onClick={() => setNotesOpen(!notesOpen)}
                className="w-full flex items-center justify-between px-5 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors rounded-xl"
              >
                <span>Notes</span>
                <div className="flex items-center gap-2">
                  {notesSaved && (
                    <span className="text-xs text-emerald-600">Saved</span>
                  )}
                  {notesOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </div>
              </button>
              {notesOpen && (
                <div className="px-5 pb-4">
                  <textarea
                    value={notes}
                    onChange={(e) => handleNotesChange(e.target.value)}
                    placeholder="Write your notes about this paper..."
                    className="w-full h-32 text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-y"
                  />
                </div>
              )}
            </div>

            {/* QA Section */}
            {isCompleted && (
              <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-3">
                <h3 className="text-sm font-semibold text-gray-900">Ask Questions</h3>

                {/* Messages */}
                {sessionDetail.messages?.length > 0 && (
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {sessionDetail.messages.map((msg: DeepReadMessage) => (
                      <div
                        key={msg.id}
                        className={`rounded-lg px-4 py-3 text-sm ${
                          msg.role === "user"
                            ? "bg-blue-50 text-blue-900 ml-8"
                            : "bg-gray-50 text-gray-800 mr-8"
                        }`}
                      >
                        {msg.status === "pending" || msg.status === "generating" ? (
                          <span className="flex items-center gap-2 text-gray-400">
                            <Loader2 size={14} className="animate-spin" />
                            {msg.status === "generating" ? "Thinking..." : "Queued..."}
                          </span>
                        ) : msg.content ? (
                          <MathMarkdown className="prose prose-sm max-w-none">
                            {msg.content}
                          </MathMarkdown>
                        ) : (
                          <span className="text-gray-400 italic">
                            {msg.status === "failed" ? "Failed to generate response" : ""}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Input */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={qaInput}
                    onChange={(e) => setQaInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSendQA();
                      }
                    }}
                    placeholder="Ask about the paper..."
                    disabled={!!pendingMsgId}
                    className="flex-1 px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
                  />
                  <button
                    onClick={handleSendQA}
                    disabled={!qaInput.trim() || !!pendingMsgId}
                    className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {pendingMsgId ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Send size={14} />
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Failed state */}
            {sessionDetail.status === "failed" && (
              <div className="bg-red-50 border border-red-200 rounded-xl px-5 py-4 text-sm text-red-700">
                Deep reading failed. Try starting a new session.
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function CardList({
  title,
  items,
  color,
}: {
  title: string;
  items?: string[];
  color: string;
}) {
  if (!items?.length) return null;
  const colorMap: Record<string, string> = {
    blue: "bg-blue-50 text-blue-800",
    emerald: "bg-emerald-50 text-emerald-800",
    amber: "bg-amber-50 text-amber-800",
    purple: "bg-purple-50 text-purple-800",
    rose: "bg-rose-50 text-rose-800",
  };
  const cls = colorMap[color] || colorMap.blue;
  return (
    <div>
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
        {title}
      </h4>
      <ul className="space-y-1">
        {items.map((item, i) => (
          <li
            key={i}
            className={`text-xs rounded-lg px-2.5 py-1.5 ${cls}`}
          >
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
