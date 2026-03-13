"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Plus,
  Send,
  Trash2,
  MessageSquare,
  Loader2,
  Database,
  CheckCircle2,
} from "lucide-react";
import { api, ChatMessage } from "@/lib/api";
import { MathMarkdown } from "@/components/MathMarkdown";

export default function ChatPage() {
  const { id: topicId } = useParams<{ id: string }>();
  const qc = useQueryClient();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [pendingMsgId, setPendingMsgId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // --- Embedding index status ---
  const { data: embStatus } = useQuery({
    queryKey: ["embeddingsStatus", topicId],
    queryFn: () => api.getEmbeddingsStatus(topicId),
    refetchInterval: (query) => {
      const d = query.state.data;
      return d?.progress?.status === "running" ? 2_000 : 30_000;
    },
  });

  const buildEmbeddings = useMutation({
    mutationFn: () => api.buildEmbeddings(topicId),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["embeddingsStatus", topicId] }),
  });

  // --- Queries ---
  const { data: sessionsData } = useQuery({
    queryKey: ["chatSessions", topicId],
    queryFn: () => api.listChatSessions(topicId),
    refetchInterval: 5_000,
  });
  const sessions = sessionsData?.sessions ?? [];

  const { data: sessionDetail, refetch: refetchSession } = useQuery({
    queryKey: ["chatSession", topicId, activeSessionId],
    queryFn: () => api.getChatSession(topicId, activeSessionId!),
    enabled: !!activeSessionId,
    refetchInterval: pendingMsgId ? 3_000 : false,
  });
  const messages = sessionDetail?.messages ?? [];

  // Poll assistant message progress
  const { data: progressData } = useQuery({
    queryKey: ["chatProgress", topicId, activeSessionId, pendingMsgId],
    queryFn: () =>
      api.getChatMessageProgress(topicId, activeSessionId!, pendingMsgId!),
    enabled: !!pendingMsgId && !!activeSessionId,
    refetchInterval: 2_000,
  });

  // When progress shows completed/failed, clear polling and refetch messages
  useEffect(() => {
    if (
      progressData &&
      (progressData.status === "completed" || progressData.status === "failed")
    ) {
      setPendingMsgId(null);
      refetchSession();
    }
  }, [progressData, refetchSession]);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, pendingMsgId]);

  // --- Mutations ---
  const createSession = useMutation({
    mutationFn: () => api.createChatSession(topicId),
    onSuccess: (s) => {
      qc.invalidateQueries({ queryKey: ["chatSessions", topicId] });
      setActiveSessionId(s.id);
    },
  });

  const deleteSession = useMutation({
    mutationFn: (sid: string) => api.deleteChatSession(topicId, sid),
    onSuccess: (_, sid) => {
      qc.invalidateQueries({ queryKey: ["chatSessions", topicId] });
      if (activeSessionId === sid) {
        setActiveSessionId(null);
      }
    },
  });

  const sendMessage = useMutation({
    mutationFn: (content: string) =>
      api.sendChatMessage(topicId, activeSessionId!, content),
    onSuccess: (res) => {
      setPendingMsgId(res.assistant_msg_id);
      refetchSession();
      setInput("");
    },
  });

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || !activeSessionId || sendMessage.isPending || pendingMsgId)
      return;
    sendMessage.mutate(text);
  }, [input, activeSessionId, sendMessage, pendingMsgId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleDelete = (sid: string, title: string) => {
    if (!confirm(`Delete chat "${title || "Untitled"}"?`)) return;
    deleteSession.mutate(sid);
  };

  // Auto-select first session if none active
  useEffect(() => {
    if (!activeSessionId && sessions.length > 0) {
      setActiveSessionId(sessions[0].id);
    }
  }, [activeSessionId, sessions]);

  const embIndexed = embStatus?.indexed;
  const embRunning = embStatus?.progress?.status === "running";
  const embPaperCount = embStatus?.paper_count ?? 0;
  const embEmbCount = embStatus?.embedding_count ?? 0;

  return (
    <div className="flex gap-4 min-h-[600px]">
      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 space-y-3">
        <button
          onClick={() => createSession.mutate()}
          disabled={createSession.isPending}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          <Plus size={14} />
          New Chat
        </button>

        {/* Embedding index status */}
        <div className="px-3 py-2 bg-gray-50 rounded-lg border border-gray-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-xs text-gray-500">
              <Database size={12} />
              <span>RAG Index</span>
            </div>
            {embIndexed ? (
              <CheckCircle2 size={12} className="text-green-500" />
            ) : embRunning ? (
              <Loader2 size={12} className="text-blue-500 animate-spin" />
            ) : null}
          </div>
          <p className="text-xs text-gray-400 mt-1">
            {embEmbCount}/{embPaperCount} papers indexed
          </p>
          {embRunning && embStatus?.progress && (
            <div className="mt-1.5">
              <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all"
                  style={{
                    width: `${embStatus.progress.total ? Math.round(((embStatus.progress.embedded ?? 0) / embStatus.progress.total) * 100) : 0}%`,
                  }}
                />
              </div>
            </div>
          )}
          {!embIndexed && !embRunning && embPaperCount > 0 && (
            <button
              onClick={() => buildEmbeddings.mutate()}
              disabled={buildEmbeddings.isPending}
              className="mt-1.5 text-xs text-blue-600 hover:text-blue-700 font-medium disabled:opacity-50"
            >
              Build Index
            </button>
          )}
        </div>

        <div className="space-y-1 max-h-[460px] overflow-y-auto">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                activeSessionId === s.id
                  ? "bg-blue-50 border border-blue-200"
                  : "hover:bg-gray-50 border border-transparent"
              }`}
              onClick={() => setActiveSessionId(s.id)}
            >
              <MessageSquare
                size={14}
                className={
                  activeSessionId === s.id ? "text-blue-600" : "text-gray-400"
                }
              />
              <div className="flex-1 min-w-0">
                <p className="text-sm truncate font-medium text-gray-800">
                  {s.title || "New Chat"}
                </p>
                <p className="text-xs text-gray-400">
                  {s.message_count} messages
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(s.id, s.title);
                }}
                className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-50 text-gray-400 hover:text-red-500 transition-all"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <p className="text-sm text-gray-400 text-center py-8">
              No chats yet
            </p>
          )}
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col bg-white rounded-xl border border-gray-200 overflow-hidden">
        {activeSessionId ? (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
              {messages.length === 0 && !pendingMsgId && (
                <div className="text-center text-gray-400 text-sm py-16">
                  Start a conversation about your research papers.
                </div>
              )}
              {messages.map((msg) => (
                <ChatBubble key={msg.id} message={msg} />
              ))}
              {pendingMsgId &&
                !messages.some(
                  (m) => m.id === pendingMsgId && m.status === "completed"
                ) && (
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center flex-shrink-0">
                      <Loader2
                        size={14}
                        className="text-purple-600 animate-spin"
                      />
                    </div>
                    <div className="bg-gray-50 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[80%]">
                      <p className="text-sm text-gray-500">Thinking...</p>
                    </div>
                  </div>
                )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-gray-100 px-4 py-3">
              <div className="flex items-end gap-2">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about your papers..."
                  rows={1}
                  className="flex-1 resize-none border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent max-h-32"
                  style={{
                    height: "auto",
                    minHeight: "40px",
                  }}
                  onInput={(e) => {
                    const target = e.target as HTMLTextAreaElement;
                    target.style.height = "auto";
                    target.style.height = `${Math.min(target.scrollHeight, 128)}px`;
                  }}
                />
                <button
                  onClick={handleSend}
                  disabled={
                    !input.trim() ||
                    sendMessage.isPending ||
                    !!pendingMsgId
                  }
                  className="p-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                >
                  <Send size={16} />
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
            Select or create a chat to begin.
          </div>
        )}
      </div>
    </div>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-[80%]">
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  // Assistant message
  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center flex-shrink-0 mt-1">
        <MessageSquare size={14} className="text-purple-600" />
      </div>
      <div className="max-w-[85%] space-y-2">
        <div className="bg-gray-50 rounded-2xl rounded-tl-sm px-4 py-3">
          {message.status === "generating" ? (
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <Loader2 size={14} className="animate-spin" />
              Generating...
            </div>
          ) : message.status === "failed" ? (
            <p className="text-sm text-red-500">
              {message.content || "Failed to generate response."}
            </p>
          ) : (
            <div className="text-sm prose prose-sm max-w-none prose-p:my-1 prose-headings:my-2">
              <MathMarkdown>{message.content}</MathMarkdown>
            </div>
          )}
        </div>
        {/* Cited papers */}
        {message.cited_papers && message.cited_papers.length > 0 && (
          <div className="flex flex-wrap gap-1.5 px-1">
            {message.cited_papers.map((p, i) => (
              <a
                key={i}
                href={
                  p.arxiv_id.startsWith("http")
                    ? p.arxiv_id
                    : `https://arxiv.org/abs/${p.arxiv_id}`
                }
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-blue-50 text-blue-700 rounded-md hover:bg-blue-100 transition-colors truncate max-w-[250px]"
                title={p.title}
              >
                {p.title}
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
