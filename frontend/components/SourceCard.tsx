'use client';

import { Source, formatSourceDisplay } from '@/lib/api';

interface SourceCardProps {
  source: Source;
}

/**
 * SourceCard component displays a single source citation.
 *
 * - Textbook sources show: 📖 **Title** – Chapter (pp. X–Y)
 * - Blog sources show as clickable links
 */
export function SourceCard({ source }: SourceCardProps) {
  const formatted = formatSourceDisplay(source);

  if (formatted.url) {
    // Blog/web source - render as link
    return (
      <a
        href={formatted.url}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-start gap-2 p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors border border-slate-200"
      >
        <span className="text-lg">{formatted.icon}</span>
        <span className="text-sm text-blue-600 hover:underline">
          {formatted.text}
        </span>
      </a>
    );
  }

  // Textbook source - render as static card
  return (
    <div className="flex items-start gap-2 p-3 rounded-lg bg-indigo-50 border border-indigo-100">
      <span className="text-lg">{formatted.icon}</span>
      <span className="text-sm text-slate-700">
        <strong>{source.title}</strong>
        {source.chapter && (
          <span className="text-slate-600"> – {source.chapter}</span>
        )}
        {source.page_start !== undefined && (
          <span className="text-slate-500">
            {' '}
            ({source.page_end && source.page_end !== source.page_start
              ? `pp. ${source.page_start}–${source.page_end}`
              : `p. ${source.page_start}`})
          </span>
        )}
      </span>
    </div>
  );
}

interface SourceListProps {
  sources: Source[];
}

/**
 * SourceList component displays a collapsible list of sources.
 */
export function SourceList({ sources }: SourceListProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3">
      <details className="group">
        <summary className="cursor-pointer text-sm text-slate-500 hover:text-slate-700 flex items-center gap-1">
          <span className="group-open:rotate-90 transition-transform">▶</span>
          <span>📚 Sources ({sources.length})</span>
        </summary>
        <div className="mt-2 space-y-2">
          {sources.map((source, index) => (
            <SourceCard key={index} source={source} />
          ))}
        </div>
      </details>
    </div>
  );
}
