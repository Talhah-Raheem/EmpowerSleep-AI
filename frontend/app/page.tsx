'use client';

import { useState, useRef, useEffect, FormEvent } from 'react';
import { useTheme } from 'next-themes';
import { Message, Source, streamMessage } from '@/lib/api';
import { getRandomQuestions } from '@/lib/sampleQuestions';
import { ChatMessage } from '@/components/ChatMessage';
import { SleepLoader } from '@/components/SleepLoader';
import { EmpowerLogo } from '@/components/EmpowerLogo';

/**
 * Main chat page component.
 *
 * Features:
 * - Chat interface with message bubbles
 * - Bottom input bar
 * - Demo question buttons
 * - Conversation history for multi-turn context
 * - Source citations
 * - Dark mode toggle
 */
export default function ChatPage() {
  // State
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sampleQuestions, setSampleQuestions] = useState<string[]>([]);
  const [mounted, setMounted] = useState(false);

  const { theme, setTheme } = useTheme();

  // Initialize sample questions on mount (client-side only)
  useEffect(() => {
    setSampleQuestions(getRandomQuestions(3));
    setMounted(true);
  }, []);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pendingSourcesRef = useRef<Source[]>([]);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const userScrolledUpRef = useRef(false);

  // Immediately stop auto-scroll when user scrolls up (fires before scroll event)
  const handleWheel = (e: React.WheelEvent) => {
    if (e.deltaY < 0) userScrolledUpRef.current = true;
  };

  // Resume auto-scroll when user manually scrolls back to the bottom
  const handleChatScroll = () => {
    const el = chatContainerRef.current;
    if (!el) return;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 30) {
      userScrolledUpRef.current = false;
    }
  };

  // Auto-scroll only when user is near the bottom (instant, no animation to fight)
  useEffect(() => {
    if (!userScrolledUpRef.current) {
      const el = chatContainerRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    }
  }, [messages, isLoading]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  /**
   * Core streaming logic shared by handleSubmit and handleRegenerate.
   * Assumes the user message is already in `messages` state.
   */
  const runStream = async (userMessageText: string, historySnapshot: Message[]) => {
    pendingSourcesRef.current = [];
    userScrolledUpRef.current = false;

    let firstToken = true;

    try {
      await streamMessage(
        userMessageText,
        historySnapshot,
        abortControllerRef.current!.signal,
        {
          onSources: (sources) => {
            pendingSourcesRef.current = sources;
          },
          onToken: (text) => {
            if (firstToken) {
              firstToken = false;
              setIsLoading(false);
              setIsStreaming(true);
              setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: text },
              ]);
            } else {
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                updated[updated.length - 1] = { ...last, content: last.content + text };
                return updated;
              });
            }
          },
          onDone: () => {
            setIsStreaming(false);
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = { ...last, sources: pendingSourcesRef.current };
              return updated;
            });
          },
          onError: (err) => {
            setIsStreaming(false);
            setIsLoading(false);
            setError(err.message);
            setMessages((prev) => prev.slice(0, -1));
          },
        },
      );
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return;
      setError(err instanceof Error ? err.message : 'Failed to send message');
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      abortControllerRef.current = null;
      setIsLoading(false);
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  };

  /**
   * Handle sending a message
   */
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading || isStreaming) return;

    setInput('');
    setError(null);
    if (inputRef.current) inputRef.current.style.height = 'auto';

    abortControllerRef.current = new AbortController();
    const historySnapshot = messages;

    setMessages((prev) => [...prev, { role: 'user', content: trimmedInput }]);
    setIsLoading(true);

    await runStream(trimmedInput, historySnapshot);
  };

  /**
   * Regenerate the assistant response at the given message index.
   * Removes that response and re-streams it using the same user message.
   */
  const handleRegenerate = async (assistantIndex: number) => {
    if (isLoading || isStreaming) return;

    const userMsg = messages[assistantIndex - 1];
    if (!userMsg || userMsg.role !== 'user') return;

    const historySnapshot = messages.slice(0, assistantIndex - 1);

    // Drop the old assistant response (and anything after it)
    setMessages((prev) => prev.slice(0, assistantIndex));
    setError(null);

    abortControllerRef.current = new AbortController();
    setIsLoading(true);

    await runStream(userMsg.content, historySnapshot);
  };

  /**
   * Handle clicking a demo question
   */
  const handleDemoClick = (question: string) => {
    setInput(question);
    inputRef.current?.focus();
  };

  /**
   * Handle starting a new conversation
   */
  const handleNewConversation = () => {
    // Abort any in-flight request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setMessages([]);
    setError(null);
    setIsLoading(false);
    setIsStreaming(false);
    // Show fresh sample questions
    setSampleQuestions(getRandomQuestions(3));
    inputRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-screen bg-empower-50 dark:bg-empower-900">
      {/* Header */}
      <header className="bg-white dark:bg-empower-800 border-b border-empower-100 dark:border-empower-700 px-4 py-3 flex items-center justify-between shadow-sm">
        <a
          href="https://www.empowersleep.com/"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 hover:opacity-80 transition-opacity"
        >
          <EmpowerLogo className="h-10 w-10 text-empower-500 dark:text-empower-300" />
          <div>
            <h1 className="text-lg font-heading font-semibold text-empower-700 dark:text-empower-100">EmpowerSleep</h1>
            <p className="text-xs text-empower-400 dark:text-empower-500">Sleep care, simplified</p>
          </div>
        </a>
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <button
              onClick={handleNewConversation}
              className="text-sm text-empower-500 hover:text-empower-700 dark:text-empower-400 dark:hover:text-empower-200 font-medium flex items-center gap-1 px-3 py-1.5 rounded-lg hover:bg-empower-50 dark:hover:bg-empower-700 transition-colors"
            >
              <span>New Chat</span>
            </button>
          )}
          {mounted && (
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg text-empower-500 hover:text-empower-700 hover:bg-empower-50 dark:text-empower-400 dark:hover:text-empower-200 dark:hover:bg-empower-700 transition-colors"
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? (
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" />
                  <line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" />
                  <line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
              )}
            </button>
          )}
        </div>
      </header>

      {/* Chat area */}
      <main ref={chatContainerRef} onScroll={handleChatScroll} onWheel={handleWheel} className="flex-1 overflow-y-auto chat-scrollbar px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {/* Welcome message when empty */}
          {messages.length === 0 && (
            <div className="text-center py-12 animate-fade-in">
              <EmpowerLogo className="h-20 w-20 mx-auto mb-4 text-empower-500 dark:text-empower-300" />
              <h2 className="text-2xl font-heading font-semibold text-empower-700 dark:text-empower-100 mb-2">
                Welcome to EmpowerSleep
              </h2>
              <p className="text-empower-500 dark:text-empower-400 mb-8 max-w-md mx-auto">
                Ask me anything about sleep. I&apos;ll provide educational information
                grounded in expert content.
              </p>

              {/* Sample questions - randomly selected */}
              <div className="flex flex-wrap justify-center gap-2">
                {sampleQuestions.map((question) => (
                  <button
                    key={question}
                    onClick={() => handleDemoClick(question)}
                    className="px-4 py-2 bg-white dark:bg-empower-800 border border-empower-200 dark:border-empower-700 rounded-full text-sm text-empower-600 dark:text-empower-300 hover:bg-empower-50 dark:hover:bg-empower-700 hover:border-empower-300 dark:hover:border-empower-600 hover:text-empower-700 dark:hover:text-empower-200 transition-colors shadow-sm"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((message, index) => (
            <ChatMessage
              key={index}
              message={message}
              streaming={isStreaming && index === messages.length - 1 && message.role === 'assistant'}
              onRegenerate={message.role === 'assistant' ? () => handleRegenerate(index) : undefined}
            />
          ))}

          {/* Loading indicator - branded sleep-themed loader */}
          {isLoading && <SleepLoader />}

          {/* Error message */}
          {error && (
            <div className="flex justify-center animate-fade-in">
              <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-xl px-4 py-3 text-sm text-red-600 dark:text-red-400">
                {error}
              </div>
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input area */}
      <footer className="bg-white dark:bg-empower-800 border-t border-empower-100 dark:border-empower-700 px-4 py-4 shadow-[0_-2px_10px_rgba(0,0,0,0.03)] dark:shadow-[0_-2px_10px_rgba(0,0,0,0.3)]">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="flex gap-3 items-end">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = 'auto';
                const newHeight = Math.min(e.target.scrollHeight, 96);
                e.target.style.height = newHeight + 'px';
                e.target.style.overflowY = e.target.scrollHeight > 96 ? 'auto' : 'hidden';
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (input.trim() && !isLoading && !isStreaming) {
                    handleSubmit(e as unknown as FormEvent);
                  }
                }
              }}
              placeholder="Ask me about sleep..."
              disabled={isLoading || isStreaming}
              rows={1}
              className="flex-1 px-5 py-3 rounded-2xl border border-empower-200 dark:border-empower-600 bg-white dark:bg-empower-700 text-empower-800 dark:text-empower-100 placeholder:text-empower-300 dark:placeholder:text-empower-500 focus:outline-none focus:ring-2 focus:ring-empower-400 dark:focus:ring-empower-500 focus:border-transparent disabled:bg-empower-50 dark:disabled:bg-empower-800 disabled:text-empower-300 dark:disabled:text-empower-600 transition-shadow resize-none overflow-hidden"
              style={{ maxHeight: '96px' }}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading || isStreaming}
              className="px-6 py-3 bg-empower-500 dark:bg-empower-600 text-white rounded-full font-medium hover:bg-empower-600 dark:hover:bg-empower-500 disabled:bg-empower-200 dark:disabled:bg-empower-700 disabled:cursor-not-allowed transition-colors shadow-sm"
            >
              Send
            </button>
          </div>
          <p className="text-xs text-empower-400 dark:text-empower-500 text-center mt-2">
            Educational information only. Not medical advice.
          </p>
        </form>
      </footer>
    </div>
  );
}
