"use client";

import { useQuery } from "@tanstack/react-query";
import { RefreshCw, ArrowLeft, Zap, Bot, Github } from "lucide-react";
import Link from "next/link";
import { api, ServiceUsage, UsageLimit } from "@/lib/api";

const SERVICE_META: Record<
  string,
  { label: string; icon: React.ReactNode; color: string; bg: string; ring: string; trackBg: string }
> = {
  claude: {
    label: "Claude Code",
    icon: <Zap size={20} />,
    color: "text-orange-600",
    bg: "bg-orange-500",
    ring: "ring-orange-200",
    trackBg: "bg-orange-100",
  },
  codex: {
    label: "Codex CLI",
    icon: <Bot size={20} />,
    color: "text-emerald-600",
    bg: "bg-emerald-500",
    ring: "ring-emerald-200",
    trackBg: "bg-emerald-100",
  },
  copilot: {
    label: "GitHub Copilot",
    icon: <Github size={20} />,
    color: "text-violet-600",
    bg: "bg-violet-500",
    ring: "ring-violet-200",
    trackBg: "bg-violet-100",
  },
};

export default function UsagePage() {
  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ["usage"],
    queryFn: () => api.getUsage(),
    refetchInterval: 120_000, // refresh every 2 min
  });

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Header */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-all"
              >
                <ArrowLeft size={16} />
              </Link>
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">
                  CLI Usage
                </h1>
                <p className="text-slate-400 text-sm mt-0.5">
                  Claude Code / Codex / Copilot limits & consumption
                </p>
              </div>
            </div>
            <button
              onClick={() => refetch()}
              disabled={isFetching}
              className="p-2.5 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-all disabled:opacity-50"
              title="Refresh"
            >
              <RefreshCw
                size={16}
                className={isFetching ? "animate-spin" : ""}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-5xl mx-auto px-6 py-8">
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="bg-white rounded-2xl border border-slate-200 p-6 h-64 animate-pulse"
              />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {data?.services.map((svc) => (
              <ServiceCard key={svc.service} usage={svc} />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

function ServiceCard({ usage }: { usage: ServiceUsage }) {
  const meta = SERVICE_META[usage.service] ?? {
    label: usage.service,
    icon: <Zap size={20} />,
    color: "text-gray-600",
    bg: "bg-gray-500",
    ring: "ring-gray-200",
    trackBg: "bg-gray-100",
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
      {/* Card header */}
      <div className="px-6 pt-5 pb-4 border-b border-slate-100">
        <div className="flex items-center gap-3">
          <div className={`${meta.color}`}>{meta.icon}</div>
          <div className="flex-1">
            <h3 className="font-semibold text-slate-900">{meta.label}</h3>
            {usage.plan && (
              <span className="text-xs text-slate-500 bg-slate-100 px-2 py-0.5 rounded-full">
                {usage.plan}
              </span>
            )}
          </div>
          <StatusBadge status={usage.status} />
        </div>
      </div>

      {/* Card body */}
      <div className="px-6 py-4 space-y-4 min-h-[180px]">
        {usage.status === "unconfigured" ? (
          <div className="flex items-center justify-center h-full min-h-[140px]">
            <p className="text-sm text-slate-400 text-center leading-relaxed">
              {usage.error || "Not configured"}
            </p>
          </div>
        ) : usage.status === "error" ? (
          <div className="flex items-center justify-center h-full min-h-[140px]">
            <p className="text-sm text-red-500 text-center leading-relaxed">
              {usage.error || "Failed to fetch"}
            </p>
          </div>
        ) : usage.limits.length === 0 ? (
          <div className="flex items-center justify-center h-full min-h-[140px]">
            <p className="text-sm text-slate-400">No usage data</p>
          </div>
        ) : (
          usage.limits.map((limit, i) => (
            <LimitRow key={i} limit={limit} meta={meta} />
          ))
        )}
      </div>
    </div>
  );
}

function LimitRow({
  limit,
  meta,
}: {
  limit: UsageLimit;
  meta: (typeof SERVICE_META)[string];
}) {
  const hasBar = limit.utilization >= 0;
  const pct = Math.min(limit.utilization, 100);

  // Color intensity based on utilization
  const barColor =
    pct >= 90
      ? "bg-red-500"
      : pct >= 70
        ? "bg-amber-500"
        : meta.bg;

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-xs font-medium text-slate-600">{limit.name}</span>
        {hasBar ? (
          <span className="text-xs font-semibold tabular-nums text-slate-900">
            {limit.utilization}%
          </span>
        ) : limit.value !== undefined ? (
          <span className="text-xs font-semibold tabular-nums text-slate-900">
            {formatValue(limit.value, limit.unit)}
          </span>
        ) : null}
      </div>
      {hasBar && (
        <div className={`h-2 rounded-full ${meta.trackBg} overflow-hidden`}>
          <div
            className={`h-full rounded-full ${barColor} transition-all duration-500`}
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
      {limit.resets_at && (
        <p className="text-[10px] text-slate-400 mt-1">
          Resets {formatResetTime(limit.resets_at)}
        </p>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "ok")
    return (
      <span className="text-[10px] font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
        Connected
      </span>
    );
  if (status === "unconfigured")
    return (
      <span className="text-[10px] font-medium text-slate-500 bg-slate-50 px-2 py-0.5 rounded-full border border-slate-200">
        Not Set Up
      </span>
    );
  return (
    <span className="text-[10px] font-medium text-red-600 bg-red-50 px-2 py-0.5 rounded-full border border-red-200">
      Error
    </span>
  );
}

function formatValue(value: number, unit?: string): string {
  if (unit === "USD") return `$${value.toFixed(2)}`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M ${unit ?? ""}`.trim();
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K ${unit ?? ""}`.trim();
  return `${value} ${unit ?? ""}`.trim();
}

function formatResetTime(iso: string): string {
  try {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = d.getTime() - now.getTime();
    if (diffMs <= 0) return "soon";
    const hours = Math.floor(diffMs / 3_600_000);
    const minutes = Math.floor((diffMs % 3_600_000) / 60_000);
    if (hours >= 24) {
      const days = Math.floor(hours / 24);
      return `in ${days}d ${hours % 24}h`;
    }
    if (hours > 0) return `in ${hours}h ${minutes}m`;
    return `in ${minutes}m`;
  } catch {
    return iso;
  }
}
