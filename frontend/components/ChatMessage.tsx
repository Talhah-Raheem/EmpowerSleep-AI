'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Message } from '@/lib/api';
import { SourceList } from './SourceCard';

interface ChatMessageProps {
  message: Message;
  streaming?: boolean;
  onRegenerate?: () => void;
  onFeedback?: (rating: 1 | -1) => void;
}

export function ChatMessage({ message, streaming, onRegenerate, onFeedback }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<1 | -1 | null>(null);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[75%] bg-empower-600/90 backdrop-blur-sm text-white rounded-2xl rounded-br-md shadow-sm px-4 py-3">
          <p>{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="group animate-fade-in w-full">
      <div className="relative bg-white/90 dark:bg-black/40 backdrop-blur-sm border border-white/30 dark:border-white/10 rounded-2xl shadow-sm px-5 py-4">
      {/* Copy button */}
      <button
          onClick={handleCopy}
          className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-md text-empower-300 dark:text-empower-500 hover:text-empower-600 dark:hover:text-empower-300 hover:bg-empower-100 dark:hover:bg-empower-700"
          title="Copy message"
        >
          {copied ? (
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
            </svg>
          )}
        </button>

      {/* Message content */}
      <div className="prose-chat text-empower-800 dark:text-empower-100">
        <ReactMarkdown
          components={{
            a: ({ href, children }) => (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-empower-600 dark:text-empower-400 hover:text-empower-700 dark:hover:text-empower-300 hover:underline"
              >
                {children}
              </a>
            ),
            hr: () => <hr className="my-3 border-empower-100 dark:border-empower-700" />,
            em: ({ children }) => (
              <em className="text-empower-400 dark:text-empower-500 text-sm not-italic">{children}</em>
            ),
          }}
        >
          {message.content}
        </ReactMarkdown>
        {streaming && (
          <span className="inline-block w-[2px] h-[1em] bg-empower-500 dark:bg-empower-300 ml-0.5 align-middle animate-blink" />
        )}
      </div>

      {/* Sources */}
      {message.sources && message.sources.length > 0 && (
        <SourceList sources={message.sources} />
      )}

      {/* Action buttons */}
      {!streaming && (onRegenerate || onFeedback) && (
        <div className="mt-3 flex items-center justify-between gap-2">
          {onFeedback && (
            <div className="flex items-center gap-1">
              <button
                onClick={() => {
                  if (feedback !== null) return;
                  setFeedback(1);
                  onFeedback(1);
                }}
                disabled={feedback !== null}
                title="Helpful"
                className={`p-1 rounded-md transition-colors ${
                  feedback === 1
                    ? 'text-empower-600 dark:text-empower-300'
                    : 'text-empower-300 dark:text-empower-600 hover:text-empower-600 dark:hover:text-empower-300 hover:bg-empower-100 dark:hover:bg-empower-700'
                } disabled:cursor-default`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill={feedback === 1 ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" />
                  <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
                </svg>
              </button>
              <button
                onClick={() => {
                  if (feedback !== null) return;
                  setFeedback(-1);
                  onFeedback(-1);
                }}
                disabled={feedback !== null}
                title="Not helpful"
                className={`p-1 rounded-md transition-colors ${
                  feedback === -1
                    ? 'text-empower-600 dark:text-empower-300'
                    : 'text-empower-300 dark:text-empower-600 hover:text-empower-600 dark:hover:text-empower-300 hover:bg-empower-100 dark:hover:bg-empower-700'
                } disabled:cursor-default`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill={feedback === -1 ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" />
                  <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
                </svg>
              </button>
            </div>
          )}
          {onRegenerate && (
            <button
              onClick={onRegenerate}
              title="Regenerate response"
              className="flex items-center gap-1 text-xs text-empower-400 dark:text-empower-500 hover:text-empower-600 dark:hover:text-empower-300 transition-colors ml-auto"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
              Regenerate
            </button>
          )}
        </div>
      )}
      </div>
    </div>
  );
}
