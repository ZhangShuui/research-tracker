"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { X, ChevronDown, ChevronRight, Sparkles } from "lucide-react";
import { api, Topic, TopicCreate } from "@/lib/api";

interface Props {
  topic?: Topic;
  onClose: () => void;
}

const DEFAULT_CATEGORIES = ["cs.CV", "cs.AI", "cs.LG"];

export function TopicForm({ topic, onClose }: Props) {
  const qc = useQueryClient();
  const isEdit = Boolean(topic);

  const [name, setName] = useState(topic?.name ?? "");
  const [description, setDescription] = useState(topic?.description ?? "");
  const [arxivKeywords, setArxivKeywords] = useState(
    (topic?.arxiv_keywords ?? []).join("\n")
  );
  const [arxivCategories, setArxivCategories] = useState(
    (topic?.arxiv_categories ?? DEFAULT_CATEGORIES).join(", ")
  );
  const [arxivLookback, setArxivLookback] = useState(
    topic?.arxiv_lookback_days ?? 2
  );
  const [githubKeywords, setGithubKeywords] = useState(
    (topic?.github_keywords ?? []).join("\n")
  );
  const [githubLookback, setGithubLookback] = useState(
    topic?.github_lookback_days ?? 7
  );
  const [scheduleCron, setScheduleCron] = useState(topic?.schedule_cron ?? "");
  const [showSources, setShowSources] = useState(
    Boolean(topic?.openalex_enabled || topic?.openreview_enabled)
  );
  // OpenAlex
  const [oaEnabled, setOaEnabled] = useState(topic?.openalex_enabled ?? false);
  const [oaKeywords, setOaKeywords] = useState(
    (topic?.openalex_keywords ?? []).join("\n")
  );
  const [oaLookback, setOaLookback] = useState(topic?.openalex_lookback_days ?? 7);
  const [oaVenues, setOaVenues] = useState(
    (topic?.openalex_venues ?? []).join(", ")
  );
  const [oaMaxResults, setOaMaxResults] = useState(topic?.openalex_max_results ?? 200);
  // OpenReview
  const [orEnabled, setOrEnabled] = useState(topic?.openreview_enabled ?? false);
  const [orVenues, setOrVenues] = useState(
    (topic?.openreview_venues ?? []).join(", ")
  );
  const [orKeywords, setOrKeywords] = useState(
    (topic?.openreview_keywords ?? []).join("\n")
  );
  const [orMaxResults, setOrMaxResults] = useState(
    topic?.openreview_max_results ?? 100
  );
  const [searchDateFrom, setSearchDateFrom] = useState(
    topic?.search_date_from ?? ""
  );
  const [searchDateTo, setSearchDateTo] = useState(
    topic?.search_date_to ?? ""
  );
  const [error, setError] = useState("");

  // Quick create (name only)
  const quickCreateMut = useMutation({
    mutationFn: (topicName: string) => api.quickCreateTopic(topicName),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["topics"] });
      onClose();
    },
    onError: (e: Error) => setError(e.message),
  });

  const createMut = useMutation({
    mutationFn: (body: TopicCreate) => api.createTopic(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["topics"] });
      onClose();
    },
    onError: (e: Error) => setError(e.message),
  });

  const updateMut = useMutation({
    mutationFn: (body: TopicCreate) => api.updateTopic(topic!.id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["topics"] });
      qc.invalidateQueries({ queryKey: ["topic", topic!.id] });
      onClose();
    },
    onError: (e: Error) => setError(e.message),
  });

  const isPending = quickCreateMut.isPending || createMut.isPending || updateMut.isPending;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (isEdit) {
      const body: TopicCreate = {
        name: name.trim(),
        description: description.trim(),
        arxiv_keywords: arxivKeywords.split("\n").map((s) => s.trim()).filter(Boolean),
        arxiv_categories: arxivCategories.split(",").map((s) => s.trim()).filter(Boolean),
        arxiv_lookback_days: Number(arxivLookback),
        github_keywords: githubKeywords.split("\n").map((s) => s.trim()).filter(Boolean),
        github_lookback_days: Number(githubLookback),
        schedule_cron: scheduleCron.trim(),
        enabled: true,
        // OpenAlex
        openalex_enabled: oaEnabled,
        openalex_keywords: oaKeywords.split("\n").map((s) => s.trim()).filter(Boolean),
        openalex_lookback_days: Number(oaLookback),
        openalex_venues: oaVenues.split(",").map((s) => s.trim()).filter(Boolean),
        openalex_max_results: Number(oaMaxResults),
        // OpenReview
        openreview_enabled: orEnabled,
        openreview_venues: orVenues.split(",").map((s) => s.trim()).filter(Boolean),
        openreview_keywords: orKeywords.split("\n").map((s) => s.trim()).filter(Boolean),
        openreview_max_results: Number(orMaxResults),
        // Date range override
        search_date_from: searchDateFrom,
        search_date_to: searchDateTo,
      };
      updateMut.mutate(body);
    } else {
      // Quick create: just send the name
      quickCreateMut.mutate(name.trim());
    }
  }

  // ---- Simple create form (name only) ----
  if (!isEdit) {
    return (
      <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-2xl shadow-xl w-full max-w-md">
          <div className="flex items-center justify-between p-5 border-b">
            <h2 className="font-semibold text-lg">New Topic</h2>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X size={18} />
            </button>
          </div>
          <form onSubmit={handleSubmit} className="p-5 space-y-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-gray-700">
                Research Topic
              </label>
              <input
                required
                autoFocus
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="input text-base"
                placeholder="e.g. Video Diffusion Models"
              />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Keywords, categories, and search config will be auto-generated.
              Default: arXiv papers from the past year (up to 200 results).
              You can customize all settings later from the topic page.
            </p>

            {error && (
              <p className="text-sm text-red-600 bg-red-50 rounded-lg p-3">
                {error}
              </p>
            )}

            <div className="flex justify-end gap-3 pt-1">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isPending}
                className="flex items-center gap-2 px-5 py-2.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 font-medium"
              >
                {isPending ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Generating config...
                  </>
                ) : (
                  <>
                    <Sparkles size={14} />
                    Create Topic
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  // ---- Full edit form ----
  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-lg max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-5 border-b">
          <h2 className="font-semibold text-lg">Edit Topic</h2>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <X size={18} />
          </button>
        </div>
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
              placeholder="video diffusion&#10;world model&#10;..."
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
            Leave empty to use lookback days. When set, date range applies to all sources.
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
              placeholder="video diffusion model&#10;world model&#10;..."
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
              {showSources ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
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
                          Matches venue name substrings. Use OpenAlex source names
                          (e.g. &quot;Pattern Analysis&quot; for TPAMI).
                        </p>
                      </Field>
                      <div className="grid grid-cols-2 gap-3">
                        <Field label="Lookback Days">
                          <input
                            type="number"
                            min={1}
                            max={365}
                            value={oaLookback}
                            onChange={(e) => setOaLookback(Number(e.target.value))}
                            className="input text-sm"
                          />
                        </Field>
                        <Field label="Max Results">
                          <input
                            type="number"
                            min={10}
                            max={1000}
                            value={oaMaxResults}
                            onChange={(e) => setOaMaxResults(Number(e.target.value))}
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
                          Known: iclr2024/2025, neurips2024/2025, icml2024/2025, acl2024/2025
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
                          onChange={(e) => setOrMaxResults(Number(e.target.value))}
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

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isPending}
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {isPending ? "Saving…" : "Save Changes"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

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
