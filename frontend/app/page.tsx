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
    <div className="flex flex-col h-screen bg-empower-50">
      {/* Header */}
      <header className="bg-white border-b border-empower-100 px-4 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <img
            src="/empower_sleep_logo.jpeg"
            alt="EmpowerSleep"
            className="h-10 w-10 object-contain"
          />
          <div>
            <h1 className="text-lg font-heading font-semibold text-empower-700">EmpowerSleep</h1>
            <p className="text-xs text-empower-400">Sleep care, simplified</p>
          </div>
        </div>
        {messages.length > 0 && (
          <button
            onClick={handleNewConversation}
            className="text-sm text-empower-500 hover:text-empower-700 font-medium flex items-center gap-1 px-3 py-1.5 rounded-lg hover:bg-empower-50 transition-colors"
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
              <img
                src="/empower_sleep_logo.jpeg"
                alt="EmpowerSleep"
                className="h-20 w-20 mx-auto mb-4 object-contain"
              />
              <h2 className="text-2xl font-heading font-semibold text-empower-700 mb-2">
                Welcome to EmpowerSleep
              </h2>
              <p className="text-empower-500 mb-8 max-w-md mx-auto">
                Ask me anything about sleep. I&apos;ll provide educational information
                grounded in expert content.
              </p>

              {/* Sample questions - randomly selected */}
              <div className="flex flex-wrap justify-center gap-2">
                {sampleQuestions.map((question) => (
                  <button
                    key={question}
                    onClick={() => handleDemoClick(question)}
                    className="px-4 py-2 bg-white border border-empower-200 rounded-full text-sm text-empower-600 hover:bg-empower-50 hover:border-empower-300 hover:text-empower-700 transition-colors shadow-sm"
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
              <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-600">
                {error}
              </div>
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input area */}
      <footer className="bg-white border-t border-empower-100 px-4 py-4 shadow-[0_-2px_10px_rgba(0,0,0,0.03)]">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="flex gap-3">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me about sleep..."
              disabled={isLoading}
              className="flex-1 px-5 py-3 rounded-full border border-empower-200 focus:outline-none focus:ring-2 focus:ring-empower-400 focus:border-transparent disabled:bg-empower-50 disabled:text-empower-300 transition-shadow"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 bg-empower-500 text-white rounded-full font-medium hover:bg-empower-600 disabled:bg-empower-200 disabled:cursor-not-allowed transition-colors shadow-sm"
            >
              Send
            </button>
          </div>
          <p className="text-xs text-empower-400 text-center mt-2">
            Educational information only. Not medical advice.
          </p>
        </form>
      </footer>
    </div>
  );
}
