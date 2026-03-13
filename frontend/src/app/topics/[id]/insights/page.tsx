"use client";

import { useParams } from "next/navigation";
import { InsightsPanel } from "@/components/InsightsPanel";

export default function InsightsPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="space-y-4">
      <InsightsPanel topicId={id} />
    </div>
  );
}
