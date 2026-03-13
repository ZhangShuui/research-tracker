"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Plus,
  RefreshCw,
  BookOpen,
  Cpu,
  TrendingUp,
  Activity,
  Sigma,
} from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { TopicCard } from "@/components/TopicCard";
import { TopicForm } from "@/components/TopicForm";

export default function DashboardPage() {
  const [showForm, setShowForm] = useState(false);

  const {
    data: topics,
    isLoading,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ["topics"],
    queryFn: api.getTopics,
    refetchInterval: 15_000,
  });

  const totalTopics = topics?.length ?? 0;
  const activeTopics = topics?.filter((t) => t.is_running).length ?? 0;
  const totalPapers =
    topics?.reduce((sum, t) => sum + (t.latest_session?.paper_count ?? 0), 0) ??
    0;
  const scheduledTopics =
    topics?.filter((t) => t.schedule_cron).length ?? 0;

  return (
    <main className="min-h-screen">
      {/* Dark gradient hero header */}
      <div className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900">
        {/* Decorative elements */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4wMyI+PGNpcmNsZSBjeD0iMzAiIGN5PSIzMCIgcj0iMS41Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-100" />
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />

        <div className="relative max-w-6xl mx-auto px-6 pt-10 pb-8">
          <div className="flex items-start justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-white tracking-tight">
                Paper Tracker
              </h1>
              <p className="text-slate-400 mt-1 text-sm">
                Multi-topic research dashboard
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => refetch()}
                disabled={isFetching}
                className="p-2.5 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-all disabled:opacity-50 backdrop-blur-sm"
                title="Refresh"
              >
                <RefreshCw
                  size={16}
                  className={isFetching ? "animate-spin" : ""}
                />
              </button>
              <button
                onClick={() => setShowForm(true)}
                className="flex items-center gap-2 px-4 py-2.5 bg-indigo-500 text-white rounded-lg hover:bg-indigo-400 transition-all text-sm font-medium shadow-lg shadow-indigo-500/25"
              >
                <Plus size={16} />
                New Topic
              </button>
            </div>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatCard
              icon={<BookOpen size={18} />}
              label="Topics"
              value={totalTopics}
              accent="from-blue-500/20 to-blue-600/20"
              iconColor="text-blue-400"
            />
            <StatCard
              icon={<Activity size={18} />}
              label="Active"
              value={activeTopics}
              accent="from-emerald-500/20 to-emerald-600/20"
              iconColor="text-emerald-400"
            />
            <StatCard
              icon={<TrendingUp size={18} />}
              label="Papers Found"
              value={totalPapers}
              accent="from-violet-500/20 to-violet-600/20"
              iconColor="text-violet-400"
            />
            <StatCard
              icon={<Cpu size={18} />}
              label="Scheduled"
              value={scheduledTopics}
              accent="from-amber-500/20 to-amber-600/20"
              iconColor="text-amber-400"
            />
          </div>

          {/* Discovery navigation */}
          <div className="flex gap-3 mt-4">
            <Link
              href="/discovery/trending"
              className="flex items-center gap-2 px-4 py-2.5 bg-white/10 hover:bg-white/20 text-white/90 hover:text-white rounded-lg transition-all text-sm font-medium backdrop-blur-sm border border-white/10"
            >
              <TrendingUp size={16} />
              Trending Themes
            </Link>
            <Link
              href="/discovery/math-insights"
              className="flex items-center gap-2 px-4 py-2.5 bg-white/10 hover:bg-white/20 text-white/90 hover:text-white rounded-lg transition-all text-sm font-medium backdrop-blur-sm border border-white/10"
            >
              <Sigma size={16} />
              Math Insights
            </Link>
            <Link
              href="/usage"
              className="flex items-center gap-2 px-4 py-2.5 bg-white/10 hover:bg-white/20 text-white/90 hover:text-white rounded-lg transition-all text-sm font-medium backdrop-blur-sm border border-white/10"
            >
              <Activity size={16} />
              CLI Usage
            </Link>
          </div>
        </div>
      </div>

      {/* Topic cards area */}
      <div className="max-w-6xl mx-auto px-6 -mt-1 pb-12">
        <div className="pt-6">
          {isLoading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
              {[...Array(3)].map((_, i) => (
                <div
                  key={i}
                  className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200/60 p-5 h-48 animate-pulse"
                />
              ))}
            </div>
          ) : !topics?.length ? (
            <div className="text-center py-20">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center">
                <BookOpen size={28} className="text-slate-400" />
              </div>
              <p className="text-slate-500 mb-1 font-medium">
                No topics yet
              </p>
              <p className="text-sm text-slate-400 mb-6">
                Create your first research topic to start tracking papers.
              </p>
              <button
                onClick={() => setShowForm(true)}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium shadow-md shadow-indigo-500/20"
              >
                <Plus size={16} />
                Create your first topic
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
              {topics.map((t) => (
                <TopicCard key={t.id} topic={t} />
              ))}
            </div>
          )}
        </div>
      </div>

      {showForm && <TopicForm onClose={() => setShowForm(false)} />}
    </main>
  );
}

function StatCard({
  icon,
  label,
  value,
  accent,
  iconColor,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  accent: string;
  iconColor: string;
}) {
  return (
    <div className={`rounded-xl bg-gradient-to-br ${accent} backdrop-blur-sm border border-white/10 p-4`}>
      <div className="flex items-center gap-3">
        <div className={iconColor}>{icon}</div>
        <div>
          <p className="text-2xl font-bold text-white tabular-nums">{value}</p>
          <p className="text-xs text-slate-400 font-medium">{label}</p>
        </div>
      </div>
    </div>
  );
}
