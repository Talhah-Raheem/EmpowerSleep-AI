'use client';

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

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}
    >
      <div
        className={`max-w-[85%] md:max-w-[75%] ${
          isUser
            ? 'bg-indigo-600 text-white rounded-2xl rounded-br-md'
            : 'bg-white border border-slate-200 rounded-2xl rounded-bl-md shadow-sm'
        } px-4 py-3`}
      >
        {/* Message content */}
        <div className={`prose-chat ${isUser ? 'text-white' : 'text-slate-800'}`}>
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
                    className="text-indigo-600 hover:underline"
                  >
                    {children}
                  </a>
                ),
                // Style horizontal rules (disclaimer separator)
                hr: () => <hr className="my-3 border-slate-200" />,
                // Style emphasis (disclaimer text)
                em: ({ children }) => (
                  <em className="text-slate-500 text-sm not-italic">{children}</em>
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

/**
 * LoadingMessage component displays typing indicator while waiting for response.
 */
export function LoadingMessage() {
  return (
    <div className="flex justify-start animate-fade-in">
      <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-md shadow-sm px-4 py-3">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-slate-400 rounded-full loading-dot" />
          <div className="w-2 h-2 bg-slate-400 rounded-full loading-dot" />
          <div className="w-2 h-2 bg-slate-400 rounded-full loading-dot" />
        </div>
      </div>
    </div>
  );
}
