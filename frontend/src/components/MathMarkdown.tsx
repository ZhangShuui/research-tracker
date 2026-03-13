"use client";

import { useRef, useEffect, useMemo, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import katex from "katex";
import "katex/dist/katex.min.css";
import { WandSparkles } from "lucide-react";

type MathEntry = { id: string; latex: string; display: boolean };

let _counter = 0;

/**
 * Auto-fix common LaTeX formatting issues in markdown content:
 * - Convert \[...\] to $$...$$
 * - Convert \(...\) to $...$
 * - Wrap bare LaTeX commands (outside of delimiters) in $...$
 */
function autoFixMath(content: string): string {
  if (!content) return content;

  let result = content;

  // 1. Convert \[...\] display math to $$...$$
  result = result.replace(
    /\\\[\s*\n([\s\S]*?)\n\s*\\\]/g,
    (_m, formula) => `$$\n${formula}\n$$`
  );
  // Single-line \[...\]
  result = result.replace(
    /\\\[(.+?)\\\]/g,
    (_m, formula) => `$$${formula}$$`
  );

  // 2. Convert \(...\) inline math to $...$
  result = result.replace(
    /\\\((.+?)\\\)/g,
    (_m, formula) => `$${formula}$`
  );

  // 3. Wrap bare LaTeX commands that appear outside of $ delimiters
  // Common LaTeX commands that indicate undelimited math
  const latexCmds =
    /(?<!\$)(?<![`\\])\\(mathbf|mathrm|mathcal|mathbb|frac|sqrt|sum|prod|int|lim|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega|Phi|Psi|Omega|Delta|Gamma|Lambda|Sigma|partial|nabla|infty|cdot|times|leq|geq|neq|approx|sim|propto|forall|exists|in|notin|subset|cup|cap|wedge|vee|oplus|otimes|hat|bar|tilde|vec|dot|ddot|overline|underline|text|operatorname|log|exp|min|max|arg|sup|inf|det|dim|ker|lfloor|rfloor|lceil|rceil|langle|rangle|left|right|bigl|bigr|Bigl|Bigr)(?=[{(^_])/g;

  // Find bare LaTeX sequences and wrap them
  // Only process lines that aren't already inside code blocks
  const lines = result.split("\n");
  const processed: string[] = [];
  let inCodeBlock = false;

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      inCodeBlock = !inCodeBlock;
      processed.push(line);
      continue;
    }
    if (inCodeBlock) {
      processed.push(line);
      continue;
    }

    // Check if line has bare LaTeX outside of $...$
    if (latexCmds.test(line)) {
      // Check if the LaTeX is already inside $ delimiters
      let hasBareLaTeX = false;
      let inMath = false;
      for (let i = 0; i < line.length; i++) {
        if (line[i] === "$" && line[i - 1] !== "\\") {
          inMath = !inMath;
        }
        if (
          !inMath &&
          line[i] === "\\" &&
          i + 1 < line.length &&
          /[a-zA-Z]/.test(line[i + 1])
        ) {
          // Check if this is a known LaTeX command
          const rest = line.substring(i);
          if (latexCmds.test(rest)) {
            hasBareLaTeX = true;
            break;
          }
        }
      }

      if (hasBareLaTeX) {
        // This line has bare LaTeX - wrap the entire line in $ if it looks
        // like a formula line, or leave it if mixed with text
        // For safety, just wrap individual LaTeX sequences
        let fixed = line;
        // Find sequences of LaTeX tokens not inside $...$ and wrap them
        // Simple approach: if the line is mostly LaTeX, wrap it all in $$
        const nonSpaceChars = line.replace(/\s/g, "");
        const backslashCount = (line.match(/\\/g) || []).length;
        if (backslashCount > 2 && backslashCount / nonSpaceChars.length > 0.1) {
          // Likely a formula line - wrap in display math
          fixed = `$$\n${line}\n$$`;
        }
        processed.push(fixed);
        continue;
      }
    }
    processed.push(line);
  }

  return processed.join("\n");
}

/**
 * Extract all math from content, replace with safe text placeholders.
 */
function extractMath(content: string): { text: string; entries: MathEntry[] } {
  if (!content) return { text: "", entries: [] };

  const entries: MathEntry[] = [];
  let text = content;

  // Step 1: Extract multi-line display math $$\n...\n$$
  text = text.replace(
    /\$\$\s*\n([\s\S]*?)\n\s*\$\$/g,
    (_match, formula) => {
      const id = `MTHBLK${++_counter}X`;
      entries.push({ id, latex: formula.trim(), display: true });
      return `\n\n${id}\n\n`;
    }
  );

  // Step 2: Extract single-line display math $$...$$
  text = text.replace(
    /\$\$(.+?)\$\$/g,
    (_match, formula) => {
      const id = `MTHBLK${++_counter}X`;
      entries.push({ id, latex: formula.trim(), display: true });
      return `\n\n${id}\n\n`;
    }
  );

  // Step 3: Extract inline math $...$
  text = text.replace(
    /(?<!\$)\$(?!\$)((?:[^$\\]|\\.)+?)\$(?!\$)/g,
    (_match, formula) => {
      const id = `MTHINL${++_counter}X`;
      entries.push({ id, latex: formula, display: false });
      return id;
    }
  );

  return { text, entries };
}

/**
 * ReactMarkdown wrapper that reliably renders LaTeX math.
 *
 * Supports: $$...$$, $...$, \[...\], \(...\), and bare LaTeX auto-detection.
 * Includes a "修复渲染" button for auto-fixing broken content.
 */
/**
 * Walk a DOM subtree, find text nodes containing placeholder IDs,
 * and replace them with KaTeX-rendered elements.
 */
function replacePlaceholdersWithKatex(
  root: HTMLElement,
  entries: MathEntry[]
): void {
  if (entries.length === 0) return;

  const lookup: Record<string, MathEntry> = {};
  const ids: string[] = [];
  for (const e of entries) {
    lookup[e.id] = e;
    ids.push(e.id);
  }

  const replacements: {
    node: Text;
    segments: (string | { entry: MathEntry })[];
  }[] = [];

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let textNode: Text | null;
  while ((textNode = walker.nextNode() as Text | null)) {
    const nodeText = textNode.textContent || "";
    let hasMatch = false;
    for (const id of ids) {
      if (nodeText.includes(id)) {
        hasMatch = true;
        break;
      }
    }
    if (!hasMatch) continue;

    const segments: (string | { entry: MathEntry })[] = [];
    let remaining = nodeText;
    let found = true;
    while (found && remaining.length > 0) {
      found = false;
      let earliest = -1;
      let earliestId = "";
      for (const id of ids) {
        const idx = remaining.indexOf(id);
        if (idx !== -1 && (earliest === -1 || idx < earliest)) {
          earliest = idx;
          earliestId = id;
        }
      }
      if (earliest !== -1) {
        found = true;
        if (earliest > 0) segments.push(remaining.substring(0, earliest));
        segments.push({ entry: lookup[earliestId] });
        remaining = remaining.substring(earliest + earliestId.length);
      }
    }
    if (remaining.length > 0) segments.push(remaining);
    if (segments.length > 0) replacements.push({ node: textNode, segments });
  }

  for (const { node, segments } of replacements) {
    const parent = node.parentNode;
    if (!parent) continue;

    const frag = document.createDocumentFragment();
    for (const seg of segments) {
      if (typeof seg === "string") {
        frag.appendChild(document.createTextNode(seg));
      } else {
        const { entry } = seg;
        const wrapper = document.createElement(
          entry.display ? "div" : "span"
        );
        if (entry.display) {
          wrapper.style.margin = "1em 0";
          wrapper.style.overflowX = "auto";
        }
        try {
          katex.render(entry.latex, wrapper, {
            displayMode: entry.display,
            throwOnError: false,
            trust: true,
          });
        } catch {
          wrapper.textContent = entry.latex;
        }
        frag.appendChild(wrapper);
      }
    }
    parent.replaceChild(frag, node);
  }
}

export function MathMarkdown({
  children,
  className,
  showFixButton = false,
}: {
  children: string;
  className?: string;
  showFixButton?: boolean;
}) {
  // sourceRef: hidden div where React renders markdown (React-managed)
  // displayRef: visible div where we do KaTeX replacements (unmanaged by React)
  const sourceRef = useRef<HTMLDivElement>(null);
  const displayRef = useRef<HTMLDivElement>(null);
  const [fixedContent, setFixedContent] = useState<string | null>(null);
  const [isFixed, setIsFixed] = useState(false);

  const activeContent = fixedContent ?? children ?? "";

  // First auto-fix \[...\] and \(...\) delimiters, then extract math
  const { text, entries } = useMemo(() => {
    const normalized = autoFixMath(activeContent);
    return extractMath(normalized);
  }, [activeContent]);

  // After React renders markdown into the hidden sourceRef,
  // copy its HTML to displayRef and replace placeholders with KaTeX.
  // This avoids the React reconciliation conflict (removeChild error)
  // because we never modify React-managed DOM nodes.
  useEffect(() => {
    const source = sourceRef.current;
    const display = displayRef.current;
    if (!source || !display) return;

    // Copy rendered markdown HTML to the unmanaged display div
    display.innerHTML = source.innerHTML;

    // Replace placeholder tokens with KaTeX-rendered math
    replacePlaceholdersWithKatex(display, entries);
  }, [text, entries]);

  const handleFix = useCallback(() => {
    if (isFixed) {
      setFixedContent(null);
      setIsFixed(false);
    } else {
      const fixed = autoFixMath(children || "");
      setFixedContent(fixed);
      setIsFixed(true);
    }
  }, [children, isFixed]);

  // Detect if content likely has rendering issues (bare \[ or \( or bare LaTeX)
  const hasRenderingIssues = useMemo(() => {
    const c = children || "";
    return (
      /\\\[[\s\S]*?\\\]/.test(c) ||
      /\\\([\s\S]*?\\\)/.test(c) ||
      /(?<!\$)\\(mathbf|frac|sqrt|alpha|beta|theta|phi|sum|int|mathcal|mathbb)/.test(
        c
      )
    );
  }, [children]);

  return (
    <div className="relative">
      {(showFixButton || hasRenderingIssues) && (
        <button
          onClick={handleFix}
          className={`absolute top-1 right-1 z-10 flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium rounded-md transition-colors ${
            isFixed
              ? "bg-emerald-100 text-emerald-700 hover:bg-emerald-200"
              : "bg-slate-100 text-slate-500 hover:bg-slate-200"
          }`}
          title={isFixed ? "已修复，点击还原" : "修复渲染格式"}
        >
          <WandSparkles size={10} />
          {isFixed ? "已修复" : "修复渲染"}
        </button>
      )}
      {/* Hidden: React-managed markdown rendering (never manually modified) */}
      <div ref={sourceRef} style={{ display: "none" }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      </div>
      {/* Visible: unmanaged div with KaTeX replacements (React won't reconcile) */}
      <div ref={displayRef} className={className} />
    </div>
  );
}
