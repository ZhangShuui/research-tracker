"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

interface Props {
  total: number;
  limit: number;
  offset: number;
  onChange: (offset: number) => void;
}

export function Pagination({ total, limit, offset, onChange }: Props) {
  const page = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));
  const hasPrev = offset > 0;
  const hasNext = offset + limit < total;

  if (totalPages <= 1) return null;

  return (
    <div className="flex items-center justify-between pt-4 text-sm">
      <span className="text-gray-500">
        {total} results &middot; Page {page} of {totalPages}
      </span>
      <div className="flex items-center gap-1">
        <button
          disabled={!hasPrev}
          onClick={() => onChange(Math.max(0, offset - limit))}
          className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronLeft size={16} />
        </button>
        <button
          disabled={!hasNext}
          onClick={() => onChange(offset + limit)}
          className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}
