const BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Topic {
  id: string;
  name: string;
  description: string;
  arxiv_keywords: string[];
  arxiv_categories: string[];
  arxiv_lookback_days: number;
  github_keywords: string[];
  github_lookback_days: number;
  schedule_cron: string;
  enabled: boolean;
  created_at: string;
  is_running?: boolean;
  latest_session?: Session | null;
  // OpenAlex
  openalex_enabled?: boolean;
  openalex_keywords?: string[];
  openalex_lookback_days?: number;
  openalex_venues?: string[];
  openalex_max_results?: number;
  // OpenReview
  openreview_enabled?: boolean;
  openreview_venues?: string[];
  openreview_keywords?: string[];
  openreview_max_results?: number;
  // Date range override
  search_date_from?: string;
  search_date_to?: string;
}

export interface Session {
  id: string;
  topic_id: string;
  started_at: string;
  finished_at: string | null;
  paper_count: number;
  repo_count: number;
  status: "running" | "completed" | "failed";
  report_path: string;
  insights_path: string;
  report_content?: string;
  insights_content?: string;
}

export interface Paper {
  arxiv_id: string;
  title: string;
  authors: string;
  abstract: string;
  url: string;
  published: string;
  summary: string;
  key_insight: string;
  method: string;
  contribution: string;
  math_concepts: string[];
  venue: string;
  cited_works: string[];
  quality_score: number;
  added_at: string;
  paper_id?: string;
  source?: string;
  citation_count?: number;
}

export interface PapersResponse {
  papers: Paper[];
  total: number;
  limit: number;
  offset: number;
}

export interface ReposResponse {
  repos: {
    repo_full_name: string;
    description: string;
    url: string;
    stars: number;
    pushed_at: string;
    summary: string;
    added_at: string;
  }[];
  total: number;
  limit: number;
  offset: number;
}

export interface BrainstormSession {
  id: string;
  topic_id: string;
  mode: "auto" | "user";
  user_idea: string;
  status: "running" | "completed" | "failed";
  started_at: string;
  finished_at: string | null;
  ideas_json: BrainstormIdea[] | null;
  literature_result: string;
  logic_result: string;
  code_result: string;
  run_code_verification: boolean;
}

export interface PriorArtWork {
  arxiv_id: string;
  title: string;
  relevance?: string;
  overlap?: string;
}

export interface PriorArtResult {
  prior_works: PriorArtWork[];
  similar_works: PriorArtWork[];
  maturity_level: "NASCENT" | "GROWING" | "MATURE" | "SATURATED" | "UNKNOWN";
  total_related: number;
  novelty_assessment: string;
  recommendation: "PURSUE" | "DIFFERENTIATE" | "RECONSIDER";
  recommendation_reason: string;
}

export interface BrainstormIdea {
  title: string;
  problem: string;
  motivation: string;
  method: string;
  experiment_plan: string;
  novelty_score: number;
  feasibility_score: number;
  novelty_verdict: "NOVEL" | "PARTIALLY_NOVEL" | "ALREADY_EXISTS";
  prior_art?: PriorArtResult;
  status?: "dropped";
  review?: {
    novelty?: number;
    feasibility?: number;
    clarity?: number;
    impact?: number;
    overall?: number;
    verdict?: string;
    weaknesses?: string[];
    strengths?: string[];
    revision_instructions?: string[];
  };
}

export interface ResearchPlan {
  id: string;
  topic_id: string;
  brainstorm_session_id: string;
  idea_title: string;
  idea_json: BrainstormIdea;
  status: "running" | "completed" | "failed";
  started_at: string;
  finished_at: string | null;
  introduction: string;
  related_work: string;
  methodology: string;
  experimental_design: string;
  expected_results: string;
  timeline: string;
  review: string;
  full_markdown: string;
  review_history?: { round: number; review: string; feedback?: string | null }[];
}

export interface DiscoveryReport {
  id: string;
  type: "trending" | "math";
  status: "running" | "completed" | "failed";
  started_at: string;
  finished_at: string | null;
  content: string;
  papers_json: { arxiv_id: string; title: string; source: string }[];
  paper_count: number;
  source_stats: Record<string, number>;
  quality_score: number; // -1 = not reviewed, 0-100 = scored
  quality_flags: { issue: string; severity: "low" | "medium" | "high" }[];
}

export interface QualityReview {
  quality_score: number;
  flags: { issue: string; severity: "low" | "medium" | "high" }[];
  summary: string;
}

export interface UsageLimit {
  name: string;
  utilization: number; // 0-100 percentage, -1 if not applicable
  resets_at?: string;
  value?: number;
  unit?: string;
}

export interface ServiceUsage {
  service: "claude" | "codex" | "copilot";
  status: "ok" | "error" | "unconfigured";
  error?: string;
  plan?: string;
  limits: UsageLimit[];
}

export interface ContextOptions {
  use_insights?: boolean;
  use_reports?: boolean;
  use_github?: boolean;
  use_history?: boolean;
  use_research_plans?: boolean;
  use_citations?: boolean;
  use_questions?: boolean;
  use_novelty_map?: boolean;
}

export interface ChatSession {
  id: string;
  topic_id: string;
  title: string;
  status: "active" | "archived";
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  topic_id: string;
  role: "user" | "assistant";
  content: string;
  cited_papers: { arxiv_id: string; title: string }[];
  status: "pending" | "generating" | "completed" | "failed";
  created_at: string;
}

export interface ChatSessionDetail extends ChatSession {
  messages: ChatMessage[];
}

export interface TopicCreate {
  name: string;
  description?: string;
  arxiv_keywords?: string[];
  arxiv_categories?: string[];
  arxiv_lookback_days?: number;
  github_keywords?: string[];
  github_lookback_days?: number;
  schedule_cron?: string;
  enabled?: boolean;
  // OpenAlex
  openalex_enabled?: boolean;
  openalex_keywords?: string[];
  openalex_lookback_days?: number;
  openalex_venues?: string[];
  openalex_max_results?: number;
  // OpenReview
  openreview_enabled?: boolean;
  openreview_venues?: string[];
  openreview_keywords?: string[];
  openreview_max_results?: number;
  // Date range override
  search_date_from?: string;
  search_date_to?: string;
}

export interface TopicUpdate extends Partial<TopicCreate> {}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${text}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  // --- Topics ---
  getTopics: () => req<Topic[]>("/api/topics"),

  createTopic: (body: TopicCreate) =>
    req<Topic>("/api/topics", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  quickCreateTopic: (name: string) =>
    req<Topic>("/api/topics/quick", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  getTopic: (id: string) => req<Topic>(`/api/topics/${id}`),

  updateTopic: (id: string, body: TopicUpdate) =>
    req<Topic>(`/api/topics/${id}`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),

  deleteTopic: (id: string) =>
    req<void>(`/api/topics/${id}`, { method: "DELETE" }),

  runTopic: (id: string) =>
    req<{ status: string; topic_id: string }>(`/api/topics/${id}/run`, {
      method: "POST",
    }),

  stopTopic: (id: string) =>
    req<{ cancelled: boolean; topic_id: string }>(`/api/topics/${id}/stop`, {
      method: "POST",
    }),

  getTopicProgress: (id: string) =>
    req<{
      running: boolean;
      topic_id: string;
      stage?: string;
      message?: string;
      sources_total?: number;
      sources_done?: number;
      papers_fetched?: number;
      repos_fetched?: number;
      papers_total?: number;
      papers_done?: number;
      papers_new?: number;
      repos_new?: number;
    }>(`/api/topics/${id}/progress`),

  // --- Sessions ---
  getSessions: (id: string, limit = 20, offset = 0) =>
    req<{ sessions: Session[]; limit: number; offset: number }>(
      `/api/topics/${id}/sessions?limit=${limit}&offset=${offset}`
    ),

  getSession: (topicId: string, sessionId: string) =>
    req<Session>(`/api/topics/${topicId}/sessions/${sessionId}`),

  getLatestInsights: (id: string) =>
    req<{ topic_id: string; content: string; session: Session | null }>(
      `/api/topics/${id}/insights`
    ),

  // --- Papers / Repos ---
  getPapers: (topicId: string, params?: { search?: string; venue?: string; source?: string; date_from?: string; date_to?: string; limit?: number; offset?: number }) => {
    const sp = new URLSearchParams();
    if (params?.search) sp.set("search", params.search);
    if (params?.venue) sp.set("venue", params.venue);
    if (params?.source) sp.set("source", params.source);
    if (params?.date_from) sp.set("date_from", params.date_from);
    if (params?.date_to) sp.set("date_to", params.date_to);
    if (params?.limit) sp.set("limit", String(params.limit));
    if (params?.offset) sp.set("offset", String(params.offset));
    const qs = sp.toString();
    return req<PapersResponse>(`/api/topics/${topicId}/papers${qs ? `?${qs}` : ""}`);
  },

  getPaper: (topicId: string, arxivId: string) =>
    req<Paper>(`/api/topics/${topicId}/papers/${arxivId}`),

  deletePaper: (topicId: string, arxivId: string) =>
    req<void>(`/api/topics/${topicId}/papers/${arxivId}`, { method: "DELETE" }),

  refilterPapers: (topicId: string, body: { custom_instructions?: string; min_quality?: number; auto_delete?: boolean }) =>
    req<{ status: string; topic_id: string }>(`/api/topics/${topicId}/papers/refilter`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getRefilterStatus: (topicId: string) =>
    req<{ topic_id: string; status: string; total: number; processed: number; removed: number }>(
      `/api/topics/${topicId}/papers/refilter`
    ),

  getRepos: (topicId: string, limit = 50, offset = 0) =>
    req<ReposResponse>(`/api/topics/${topicId}/repos?limit=${limit}&offset=${offset}`),

  // --- Brainstorm ---
  startBrainstorm: (topicId: string, body: {
    mode: "auto" | "user";
    user_idea?: string;
    run_code_verification?: boolean;
    context_options?: ContextOptions;
  }) =>
    req<{ status: string; session_id: string }>(`/api/topics/${topicId}/brainstorm`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listBrainstormSessions: (topicId: string) =>
    req<{ sessions: BrainstormSession[] }>(`/api/topics/${topicId}/brainstorm`),

  getBrainstormSession: (topicId: string, sessionId: string) =>
    req<BrainstormSession>(`/api/topics/${topicId}/brainstorm/${sessionId}`),

  getBrainstormProgress: (topicId: string, sessionId: string) =>
    req<{
      running: boolean;
      session_id: string;
      stage?: string;
      message?: string;
      ideas_count?: number;
      round?: number;
      total_rounds?: number;
      accepted?: number;
    }>(`/api/topics/${topicId}/brainstorm/${sessionId}/progress`),

  checkPriorArt: (topicId: string, sessionId: string, ideaIndex: number) =>
    req<PriorArtResult>(
      `/api/topics/${topicId}/brainstorm/${sessionId}/prior-art`,
      { method: "POST", body: JSON.stringify({ idea_index: ideaIndex }) }
    ),

  // --- Research Plans ---
  startResearchPlan: (topicId: string, body: { idea: Record<string, unknown>; brainstorm_session_id?: string }) =>
    req<{ status: string; plan_id: string }>(`/api/topics/${topicId}/research-plan`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listResearchPlans: (topicId: string) =>
    req<{ plans: ResearchPlan[] }>(`/api/topics/${topicId}/research-plan`),

  getResearchPlan: (topicId: string, planId: string) =>
    req<ResearchPlan>(`/api/topics/${topicId}/research-plan/${planId}`),

  getResearchPlanProgress: (topicId: string, planId: string) =>
    req<{
      plan_id: string;
      status: string;
      sections_done: number;
      sections_total: number;
      total_chars: number;
      sections: Record<string, { done: boolean; chars: number }>;
    }>(`/api/topics/${topicId}/research-plan/${planId}/progress`),

  refineResearchPlan: (topicId: string, planId: string, body: { feedback?: string; sections?: string[] }) =>
    req<{ status: string; plan_id: string }>(
      `/api/topics/${topicId}/research-plan/${planId}/refine`,
      { method: "POST", body: JSON.stringify(body) }
    ),

  // --- Discovery ---
  startDiscovery: (type: "trending" | "math" | "community", opts?: {
    categories?: string[];
    wildcard_categories?: string[];
    lookback_days?: number;
    max_recent?: number;
    max_historical?: number;
    max_wildcard?: number;
    sample_size?: number;
    keywords?: string[];
    platforms?: string[];
    max_results_per_platform?: number;
  }) =>
    req<{ status: string; type: string }>("/api/discovery", {
      method: "POST",
      body: JSON.stringify({ type, ...opts }),
    }),

  listDiscoveryReports: (type?: "trending" | "math" | "community") =>
    req<{ reports: DiscoveryReport[] }>(
      `/api/discovery${type ? `?type=${type}` : ""}`
    ),

  getDiscoveryReport: (reportId: string) =>
    req<DiscoveryReport>(`/api/discovery/${reportId}`),

  getLatestDiscovery: (type: "trending" | "math") =>
    req<DiscoveryReport>(`/api/discovery/latest/${type}`),

  reviewDiscovery: (reportId: string) =>
    req<QualityReview>(`/api/discovery/${reportId}/review`, { method: "POST" }),

  regenerateDiscovery: (reportId: string) =>
    req<{ status: string; type: string; replacing: string }>(
      `/api/discovery/${reportId}/regenerate`,
      { method: "POST" }
    ),

  // --- Translation ---
  translate: (body: {
    source_type: string;
    source_id: string;
    field: string;
    content: string;
    language?: string;
  }) =>
    req<{ translated: string }>("/api/translate", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getTranslation: (
    sourceType: string,
    sourceId: string,
    field: string,
    language = "zh"
  ) =>
    req<{ translated: string }>(
      `/api/translate?source_type=${encodeURIComponent(sourceType)}&source_id=${encodeURIComponent(sourceId)}&field=${encodeURIComponent(field)}&language=${encodeURIComponent(language)}`
    ),

  // --- Embeddings ---
  buildEmbeddings: (topicId: string) =>
    req<{ status: string; topic_id: string }>(`/api/topics/${topicId}/embeddings`, {
      method: "POST",
    }),

  getEmbeddingsStatus: (topicId: string) =>
    req<{
      topic_id: string;
      paper_count: number;
      embedding_count: number;
      indexed: boolean;
      progress: { status?: string; embedded?: number; total?: number };
    }>(`/api/topics/${topicId}/embeddings`),

  // --- Chat ---
  createChatSession: (topicId: string, title = "") =>
    req<ChatSession>(`/api/topics/${topicId}/chat`, {
      method: "POST",
      body: JSON.stringify({ title }),
    }),

  listChatSessions: (topicId: string) =>
    req<{ sessions: ChatSession[] }>(`/api/topics/${topicId}/chat`),

  getChatSession: (topicId: string, sessionId: string) =>
    req<ChatSessionDetail>(`/api/topics/${topicId}/chat/${sessionId}`),

  deleteChatSession: (topicId: string, sessionId: string) =>
    req<void>(`/api/topics/${topicId}/chat/${sessionId}`, { method: "DELETE" }),

  sendChatMessage: (topicId: string, sessionId: string, content: string) =>
    req<{ user_msg_id: string; assistant_msg_id: string }>(
      `/api/topics/${topicId}/chat/${sessionId}/messages`,
      { method: "POST", body: JSON.stringify({ content }) }
    ),

  getChatMessageProgress: (topicId: string, sessionId: string, msgId: string) =>
    req<{ msg_id: string; status: string }>(
      `/api/topics/${topicId}/chat/${sessionId}/messages/${msgId}/progress`
    ),

  // --- Usage ---
  getUsage: (service?: string) =>
    req<{ services: ServiceUsage[] }>(
      `/api/usage${service ? `?service=${service}` : ""}`
    ),
};
