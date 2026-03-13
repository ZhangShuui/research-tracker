"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowLeft, TrendingUp, Sigma, MessageCircle } from "lucide-react";

const TABS = [
  { label: "Trending Themes", href: "/discovery/trending", icon: TrendingUp },
  { label: "Math Insights", href: "/discovery/math-insights", icon: Sigma },
  { label: "Community Ideas", href: "/discovery/community", icon: MessageCircle },
] as const;

export default function DiscoveryLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <main className="min-h-screen">
      {/* Header */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-purple-900">
        <div className="max-w-5xl mx-auto px-6 pt-8 pb-6">
          <div className="flex items-center gap-3 mb-4">
            <Link
              href="/"
              className="p-2 rounded-lg hover:bg-white/10 text-white/70 hover:text-white transition-colors"
            >
              <ArrowLeft size={16} />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                Discovery
              </h1>
              <p className="text-slate-400 text-sm mt-0.5">
                Cross-topic research insights and trends
              </p>
            </div>
          </div>

          {/* Tabs */}
          <nav className="flex gap-1">
            {TABS.map((tab) => {
              const isActive = pathname === tab.href;
              const Icon = tab.icon;
              return (
                <Link
                  key={tab.href}
                  href={tab.href}
                  className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-colors ${
                    isActive
                      ? "bg-white text-gray-900"
                      : "text-white/60 hover:text-white hover:bg-white/10"
                  }`}
                >
                  <Icon size={16} />
                  {tab.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-5xl mx-auto px-6 py-6">{children}</div>
    </main>
  );
}
