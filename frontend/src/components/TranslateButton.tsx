"use client";

import { useState, useEffect, useCallback } from "react";
import { Languages, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { MathMarkdown } from "./MathMarkdown";

interface Props {
  sourceType: string;
  sourceId: string;
  field: string;
  content: string;
  language?: string;
  onTranslated?: (translated: string) => void;
}

export function TranslateButton({
  sourceType,
  sourceId,
  field,
  content,
  language = "zh",
  onTranslated,
}: Props) {
  const [translated, setTranslated] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showChinese, setShowChinese] = useState(false);
  const [checkedCache, setCheckedCache] = useState(false);

  // Check cache on mount / when source changes
  useEffect(() => {
    if (!sourceId || !content) return;
    let cancelled = false;
    setCheckedCache(false);
    setTranslated(null);
    setShowChinese(false);

    api
      .getTranslation(sourceType, sourceId, field, language)
      .then((res) => {
        if (!cancelled && res.translated) {
          setTranslated(res.translated);
        }
      })
      .catch(() => {
        // no cache, that's fine
      })
      .finally(() => {
        if (!cancelled) setCheckedCache(true);
      });

    return () => {
      cancelled = true;
    };
  }, [sourceType, sourceId, field, language, content]);

  const handleTranslate = useCallback(async () => {
    if (translated) {
      setShowChinese(!showChinese);
      if (!showChinese) onTranslated?.(translated);
      return;
    }

    setLoading(true);
    try {
      const res = await api.translate({
        source_type: sourceType,
        source_id: sourceId,
        field,
        content,
        language,
      });
      setTranslated(res.translated);
      setShowChinese(true);
      onTranslated?.(res.translated);
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  }, [translated, showChinese, sourceType, sourceId, field, content, language, onTranslated]);

  if (!content || !checkedCache) return null;

  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        handleTranslate();
      }}
      disabled={loading}
      className="flex items-center gap-1 px-2 py-1 text-xs font-medium bg-slate-100 text-slate-600 rounded-md hover:bg-slate-200 transition-colors disabled:opacity-50 whitespace-nowrap"
      title={translated ? (showChinese ? "Show original" : "Show Chinese") : "Translate to Chinese"}
    >
      {loading ? (
        <Loader2 size={12} className="animate-spin" />
      ) : (
        <Languages size={12} />
      )}
      {loading
        ? "翻译中..."
        : translated
        ? showChinese
          ? "EN"
          : "中"
        : "翻译"}
    </button>
  );
}

/** Renders translation result below original content with markdown + LaTeX support */
export function TranslatedBlock({ content }: { content: string }) {
  return (
    <MathMarkdown className="mt-3 p-3 bg-slate-50 rounded-lg border border-slate-200 text-sm text-gray-700 leading-relaxed prose prose-sm prose-slate max-w-none">
      {content}
    </MathMarkdown>
  );
}
