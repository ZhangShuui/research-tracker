"use client";

import { useState, useEffect, useMemo } from "react";
import { useParams, useSearchParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { FileText, ChevronDown } from "lucide-react";
import { api, ResearchPlan } from "@/lib/api";
import { ResearchPlanToolbar } from "@/components/ResearchPlanToolbar";
import { ResearchPlanViewer } from "@/components/ResearchPlanViewer";
import { ResearchPlanCard } from "@/components/ResearchPlanCard";

const INITIAL_VISIBLE = 5;

export default function ResearchPlanPage() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const qc = useQueryClient();

  const [selectedPlanId, setSelectedPlanId] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  // Check for prefill from brainstorm navigation
  const fromIdea = searchParams.get("from_idea");
  const prefillBsId = searchParams.get("bs_id");
  const [prefillIdea, setPrefillIdea] = useState<Record<string, unknown> | null>(null);
  const [autoGenerate, setAutoGenerate] = useState(false);

  useEffect(() => {
    if (fromIdea) {
      try {
        const raw = sessionStorage.getItem("prefill-idea");
        if (raw) {
          const idea = JSON.parse(raw);
          setPrefillIdea(idea);
          setAutoGenerate(true);
          sessionStorage.removeItem("prefill-idea");
        }
      } catch {
        // ignore invalid data
      }
    }
  }, [fromIdea]);

  // Fetch brainstorm sessions for toolbar dropdown
  const { data: bsData } = useQuery({
    queryKey: ["brainstorm-sessions", id],
    queryFn: () => api.listBrainstormSessions(id),
  });

  // Fetch research plans
  const { data: plansData } = useQuery({
    queryKey: ["research-plans", id],
    queryFn: () => api.listResearchPlans(id),
    refetchInterval: 3_000,
  });

  // Sort plans newest first
  const plans = useMemo(() => {
    const raw = plansData?.plans ?? [];
    return [...raw].sort((a, b) => {
      const dateA = a.started_at ? new Date(a.started_at).getTime() : 0;
      const dateB = b.started_at ? new Date(b.started_at).getTime() : 0;
      return dateB - dateA;
    });
  }, [plansData]);

  const selectedPlan =
    plans.find((p) => p.id === selectedPlanId) ?? plans[0] ?? null;
  const hasRunning = plans.some((p) => p.status === "running");

  // Determine visible plans: show first INITIAL_VISIBLE unless expanded
  const visiblePlans = useMemo(() => {
    if (showAll) return plans;
    return plans.slice(0, INITIAL_VISIBLE);
  }, [plans, showAll]);

  const hiddenCount = plans.length - INITIAL_VISIBLE;

  // Group visible plans by date
  const visibleGrouped = useMemo(() => {
    const groupMap = new Map<string, ResearchPlan[]>();

    for (const p of visiblePlans) {
      const dateKey = p.started_at
        ? new Date(p.started_at).toLocaleDateString(undefined, {
            month: "short",
            day: "numeric",
            year: "numeric",
          })
        : "Unknown";
      if (!groupMap.has(dateKey)) {
        groupMap.set(dateKey, []);
      }
      groupMap.get(dateKey)!.push(p);
    }

    return Array.from(groupMap.entries()).map(([label, plansInGroup]) => ({
      label,
      plans: plansInGroup,
    }));
  }, [visiblePlans]);

  // Fetch full plan details
  const { data: planDetail } = useQuery({
    queryKey: ["research-plan-detail", id, selectedPlan?.id],
    queryFn: () => api.getResearchPlan(id, selectedPlan!.id),
    enabled: Boolean(selectedPlan),
    refetchInterval: selectedPlan?.status === "running" ? 3_000 : false,
  });

  const generateMut = useMutation({
    mutationFn: (params: {
      idea: Record<string, unknown>;
      brainstorm_session_id?: string;
    }) =>
      api.startResearchPlan(id, {
        idea: params.idea,
        brainstorm_session_id: params.brainstorm_session_id,
      }),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["research-plans", id] });
      setSelectedPlanId(data.plan_id);
    },
  });

  const refineMut = useMutation({
    mutationFn: (params: { feedback: string; sections?: string[] }) =>
      api.refineResearchPlan(id, selectedPlan!.id, params),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["research-plans", id] });
      qc.invalidateQueries({ queryKey: ["research-plan-detail", id, selectedPlan?.id] });
    },
  });

  // Auto-generate when navigating from brainstorm
  useEffect(() => {
    if (autoGenerate && prefillIdea && !generateMut.isPending) {
      setAutoGenerate(false);
      generateMut.mutate({
        idea: prefillIdea,
        brainstorm_session_id: prefillBsId || undefined,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoGenerate, prefillIdea]);

  function handleGenerate(
    idea: Record<string, unknown>,
    brainstormSessionId?: string
  ) {
    generateMut.mutate({ idea, brainstorm_session_id: brainstormSessionId });
  }

  function handleRefine(feedback: string, sections?: string[]) {
    refineMut.mutate({ feedback, sections });
  }

  return (
    <div className="space-y-6">
      {/* Toolbar */}
      <ResearchPlanToolbar
        brainstormSessions={bsData?.sessions ?? []}
        onGenerate={handleGenerate}
        isPending={generateMut.isPending || hasRunning}
        prefillIdea={prefillIdea as never}
      />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Plan history sidebar */}
        <div className="lg:col-span-1">
          <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200/60 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-700">
                Plan History
              </h3>
              <span className="text-[10px] text-slate-400 font-medium tabular-nums">
                {plans.length} plans
              </span>
            </div>

            <div className="p-3 space-y-3 max-h-[calc(100vh-20rem)] overflow-y-auto scrollbar-thin">
              {plans.length === 0 ? (
                <p className="text-sm text-slate-400 py-6 text-center">
                  No research plans yet.
                </p>
              ) : (
                <>
                  {visibleGrouped.map((group) => (
                    <div key={group.label}>
                      <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider px-1 mb-1.5">
                        {group.label}
                      </p>
                      <div className="space-y-1.5">
                        {group.plans.map((p) => (
                          <ResearchPlanCard
                            key={p.id}
                            plan={p}
                            onClick={() => setSelectedPlanId(p.id)}
                            isSelected={selectedPlan?.id === p.id}
                          />
                        ))}
                      </div>
                    </div>
                  ))}

                  {/* Show older button */}
                  {!showAll && hiddenCount > 0 && (
                    <button
                      onClick={() => setShowAll(true)}
                      className="w-full flex items-center justify-center gap-1.5 py-2 text-xs text-slate-400 hover:text-indigo-600 hover:bg-indigo-50/50 rounded-lg transition-colors font-medium"
                    >
                      <ChevronDown size={13} />
                      Show {hiddenCount} older plan{hiddenCount > 1 ? "s" : ""}
                    </button>
                  )}

                  {showAll && hiddenCount > 0 && (
                    <button
                      onClick={() => setShowAll(false)}
                      className="w-full flex items-center justify-center gap-1.5 py-2 text-xs text-slate-400 hover:text-indigo-600 hover:bg-indigo-50/50 rounded-lg transition-colors font-medium"
                    >
                      Show less
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Plan viewer */}
        <div className="lg:col-span-3">
          {!planDetail ? (
            <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200/60 shadow-sm text-center py-16">
              <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center">
                <FileText size={24} className="text-slate-400" />
              </div>
              <p className="text-slate-500 text-sm font-medium">
                Generate a research plan to get started.
              </p>
              <p className="text-slate-400 text-xs mt-1.5 max-w-sm mx-auto">
                Select an idea from brainstorm or enter a custom idea, then
                generate a publication-quality research plan with peer review
                simulation.
              </p>
            </div>
          ) : (
            <ResearchPlanViewer
              plan={planDetail}
              onRefine={planDetail.status === "completed" ? handleRefine : undefined}
              isRefining={refineMut.isPending || hasRunning}
            />
          )}
        </div>
      </div>
    </div>
  );
}
