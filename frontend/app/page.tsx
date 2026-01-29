'use client';

import { useState, useRef, useEffect, FormEvent } from 'react';
import { Message, sendMessage } from '@/lib/api';
import { getRandomQuestions } from '@/lib/sampleQuestions';
import { ChatMessage } from '@/components/ChatMessage';
import { SleepLoader } from '@/components/SleepLoader';

/**
 * Main chat page component.
 *
 * Features:
 * - Chat interface with message bubbles
 * - Bottom input bar
 * - Demo question buttons
 * - Conversation history for multi-turn context
 * - Source citations
 */
export default function ChatPage() {
  // State
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sampleQuestions, setSampleQuestions] = useState<string[]>([]);

  // Initialize sample questions on mount (client-side only)
  useEffect(() => {
    setSampleQuestions(getRandomQuestions(3));
  }, []);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  /**
   * Handle sending a message
   */
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;

    // Clear input and error
    setInput('');
    setError(null);

    // Create new AbortController for this request
    abortControllerRef.current = new AbortController();

    // Add user message
    const userMessage: Message = { role: 'user', content: trimmedInput };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send to API with conversation history and abort signal
      const response = await sendMessage(
        trimmedInput,
        messages,
        abortControllerRef.current.signal
      );

      // Add assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      // Don't show error if request was aborted (user clicked New Chat)
      if (err instanceof Error && err.name === 'AbortError') {
        return;
      }
      setError(err instanceof Error ? err.message : 'Failed to send message');
      // Remove the user message if the request failed
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      abortControllerRef.current = null;
      setIsLoading(false);
      inputRef.current?.focus();
    }
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
    // Show fresh sample questions
    setSampleQuestions(getRandomQuestions(3));
    inputRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🌙</span>
          <div>
            <h1 className="text-lg font-semibold text-slate-800">EmpowerSleep</h1>
            <p className="text-xs text-slate-500">Sleep Education Assistant</p>
          </div>
        </div>
        {messages.length > 0 && (
          <button
            onClick={handleNewConversation}
            className="text-sm text-slate-500 hover:text-slate-700 flex items-center gap-1"
          >
            <span>New Chat</span>
          </button>
        )}
      </header>

      {/* Chat area */}
      <main className="flex-1 overflow-y-auto chat-scrollbar px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {/* Welcome message when empty */}
          {messages.length === 0 && (
            <div className="text-center py-12 animate-fade-in">
              <div className="text-5xl mb-4">🌙</div>
              <h2 className="text-xl font-semibold text-slate-800 mb-2">
                Welcome to EmpowerSleep
              </h2>
              <p className="text-slate-600 mb-8 max-w-md mx-auto">
                Ask me anything about sleep. I&apos;ll provide educational information
                grounded in expert content.
              </p>

              {/* Sample questions - randomly selected */}
              <div className="flex flex-wrap justify-center gap-2">
                {sampleQuestions.map((question) => (
                  <button
                    key={question}
                    onClick={() => handleDemoClick(question)}
                    className="px-4 py-2 bg-white border border-slate-200 rounded-full text-sm text-slate-600 hover:bg-slate-50 hover:border-slate-300 transition-colors"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((message, index) => (
            <ChatMessage key={index} message={message} />
          ))}

          {/* Loading indicator - branded sleep-themed loader */}
          {isLoading && <SleepLoader />}

          {/* Error message */}
          {error && (
            <div className="flex justify-center animate-fade-in">
              <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-600">
                ⚠️ {error}
              </div>
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input area */}
      <footer className="bg-white border-t border-slate-200 px-4 py-3">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="flex gap-3">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me about sleep..."
              disabled={isLoading}
              className="flex-1 px-4 py-3 rounded-full border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-slate-50 disabled:text-slate-400"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 bg-indigo-600 text-white rounded-full font-medium hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
            >
              Send
            </button>
          </div>
          <p className="text-xs text-slate-400 text-center mt-2">
            Educational information only. Not medical advice.
          </p>
        </form>
      </footer>
    </div>
  );
}
