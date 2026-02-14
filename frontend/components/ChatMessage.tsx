'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Message } from '@/lib/api';
import { SourceList } from './SourceCard';

interface ChatMessageProps {
  message: Message;
}

/**
 * ChatMessage component displays a single chat message with appropriate styling.
 *
 * - User messages: right-aligned, blue bubble
 * - Assistant messages: left-aligned, gray bubble with sources
 */
export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}
    >
      <div
        className={`relative max-w-[85%] md:max-w-[75%] ${
          isUser
            ? 'bg-empower-500 text-white rounded-2xl rounded-br-md shadow-sm'
            : 'group bg-white dark:bg-empower-800 border border-empower-100 dark:border-empower-700 rounded-2xl rounded-bl-md shadow-sm'
        } px-4 py-3 ${!isUser ? 'pr-9' : ''}`}
      >
        {/* Copy button (assistant messages only) */}
        {!isUser && (
          <button
            onClick={handleCopy}
            className="absolute top-2 right-2 p-1 rounded-md opacity-0 group-hover:opacity-100 transition-opacity text-empower-300 dark:text-empower-500 hover:text-empower-600 dark:hover:text-empower-300 hover:bg-empower-50 dark:hover:bg-empower-700"
            title="Copy message"
          >
            {copied ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
            )}
          </button>
        )}

        {/* Message content */}
        <div className={`prose-chat ${isUser ? 'text-white' : 'text-empower-800 dark:text-empower-100'}`}>
          {isUser ? (
            // User messages - plain text
            <p>{message.content}</p>
          ) : (
            // Assistant messages - render markdown
            <ReactMarkdown
              components={{
                // Style links
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
                // Style horizontal rules (disclaimer separator)
                hr: () => <hr className="my-3 border-empower-100 dark:border-empower-700" />,
                // Style emphasis (disclaimer text)
                em: ({ children }) => (
                  <em className="text-empower-400 dark:text-empower-500 text-sm not-italic">{children}</em>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Sources (assistant messages only) */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <SourceList sources={message.sources} />
        )}
      </div>
    </div>
  );
}
