'use client';

import { useState, useRef, useEffect, FormEvent } from 'react';
import { useTheme } from 'next-themes';
import { Message, MessageAttachment, Source, streamMessage, submitFeedback, getSuggestions } from '@/lib/api';
import { getRandomQuestions } from '@/lib/sampleQuestions'
import { trackEvent } from '@/lib/analytics';
import { ChatMessage } from '@/components/ChatMessage';
import { SleepLoader } from '@/components/SleepLoader';
import { EmpowerLogo } from '@/components/EmpowerLogo';
import { StarField } from '@/components/StarField';

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
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);

  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [filePreviewUrls, setFilePreviewUrls] = useState<(string | null)[]>([]);

  const { theme, setTheme } = useTheme();

  // Initialize sample questions on mount (client-side only)
  useEffect(() => {
    setSampleQuestions(getRandomQuestions(3));
    setMounted(true);
  }, []);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pendingSourcesRef = useRef<Source[]>([]);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const userScrolledUpRef = useRef(false);
  const sessionIdRef = useRef<string>(crypto.randomUUID());
  const assistantAddedRef = useRef(false);
  const suggestionsAbortRef = useRef<AbortController | null>(null);

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

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files ?? []);
    if (!selected.length) return;
    const remaining = Math.max(0, 5 - attachedFiles.length);
    const toAdd = selected.slice(0, remaining);
    const newUrls = toAdd.map((f) => (f.type.startsWith('image/') ? URL.createObjectURL(f) : null));
    setAttachedFiles((prev) => [...prev, ...toAdd]);
    setFilePreviewUrls((prev) => [...prev, ...newUrls]);
    e.target.value = ''; // reset so same file can be re-selected
  };

  const removeFile = (index: number) => {
    const url = filePreviewUrls[index];
    if (url) URL.revokeObjectURL(url);
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
    setFilePreviewUrls((prev) => prev.filter((_, i) => i !== index));
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const imageItems = Array.from(e.clipboardData.items).filter((item) =>
      item.type.startsWith('image/'),
    );
    if (!imageItems.length) return;
    const remaining = Math.max(0, 5 - attachedFiles.length);
    imageItems.slice(0, remaining).forEach((item) => {
      const file = item.getAsFile();
      if (!file) return;
      const ext = item.type.split('/')[1] || 'png';
      const named = new File([file], `pasted-image-${Date.now()}.${ext}`, { type: item.type });
      const url = URL.createObjectURL(named);
      setAttachedFiles((prev) => [...prev, named]);
      setFilePreviewUrls((prev) => [...prev, url]);
    });
    // Only prevent default if we captured images (let text paste through normally)
    if (imageItems.length > 0) e.preventDefault();
  };

  /**
   * Core streaming logic shared by handleSubmit and handleRegenerate.
   * Assumes the user message is already in `messages` state.
   */
  const runStream = async (userMessageText: string, historySnapshot: Message[], files?: File[]) => {
    pendingSourcesRef.current = [];
    userScrolledUpRef.current = false;
    assistantAddedRef.current = false;
    suggestionsAbortRef.current?.abort();
    setSuggestions([]);

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
              assistantAddedRef.current = true;
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
            // Fetch follow-up suggestions, cancellable if user starts a new chat
            const abortController = new AbortController();
            suggestionsAbortRef.current = abortController;
            getSuggestions(userMessageText, '', historySnapshot, abortController.signal).then(setSuggestions);
          },
          onError: (err) => {
            setIsStreaming(false);
            setIsLoading(false);
            setError(err.message);
            trackEvent('stream_error', { error: err.message });
            // Only remove the last message if the assistant message was actually added
            if (assistantAddedRef.current) {
              setMessages((prev) => prev.slice(0, -1));
            }
            assistantAddedRef.current = false;
          },
        },
        files,
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
    if ((!trimmedInput && attachedFiles.length === 0) || isLoading || isStreaming) return;

    setInput('');
    setError(null);
    if (inputRef.current) inputRef.current.style.height = 'auto';

    // Snapshot and clear files before async work
    const filesToSend = [...attachedFiles];
    const urlsSnapshot = [...filePreviewUrls];
    setAttachedFiles([]);
    setFilePreviewUrls([]);

    abortControllerRef.current = new AbortController();
    const historySnapshot = messages;

    // Build attachment metadata for message display (rendered in Segment 3)
    const attachments: MessageAttachment[] | undefined = filesToSend.length > 0
      ? filesToSend.map((f, i) => ({
          type: (f.type.startsWith('image/') ? 'image' : 'pdf') as 'image' | 'pdf',
          name: f.name,
          url: urlsSnapshot[i] ?? undefined,
        }))
      : undefined;

    setMessages((prev) => [...prev, { role: 'user', content: trimmedInput, ...(attachments && { attachments }) }]);
    setIsLoading(true);
    trackEvent('question_asked', { message_length: trimmedInput.length, has_history: messages.length > 0, has_files: filesToSend.length > 0 });

    await runStream(trimmedInput, historySnapshot, filesToSend);
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
   * Submit thumbs up/down feedback for an assistant message.
   */
  const handleFeedback = async (messageIndex: number, rating: 1 | -1) => {
    const aiMessage = messages[messageIndex];
    const userMessage = messages[messageIndex - 1];
    if (!aiMessage || !userMessage) return;
    await submitFeedback(
      sessionIdRef.current,
      userMessage.content,
      aiMessage.content,
      rating,
    );
  };

  /**
   * Handle clicking a demo question
   */
  const handleDemoClick = (question: string) => {
    setInput(question);
    inputRef.current?.focus();
    trackEvent('sample_question_clicked', { question });
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
    suggestionsAbortRef.current?.abort();
    suggestionsAbortRef.current = null;
    // Revoke any pending file preview URLs
    filePreviewUrls.forEach((url) => { if (url) URL.revokeObjectURL(url); });
    setAttachedFiles([]);
    setFilePreviewUrls([]);
    setMessages([]);
    setError(null);
    setIsLoading(false);
    setIsStreaming(false);
    trackEvent('new_conversation_started');
    // Show fresh sample questions
    setSampleQuestions(getRandomQuestions(3));
    inputRef.current?.focus();
  };

  const themeToggleButton = mounted && (
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
  );

  const inputForm = (
    <form onSubmit={handleSubmit}>
      {/* File preview chips */}
      {attachedFiles.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {attachedFiles.map((file, i) => (
            <div
              key={i}
              className="flex items-center gap-1.5 bg-white/20 dark:bg-white/10 backdrop-blur-sm border border-white/30 dark:border-white/20 rounded-lg px-2 py-1 text-xs text-white"
            >
              {file.type.startsWith('image/') && filePreviewUrls[i] ? (
                <img src={filePreviewUrls[i]!} alt="" className="h-5 w-5 rounded object-cover flex-shrink-0" />
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
                </svg>
              )}
              <span className="max-w-[120px] truncate">{file.name}</span>
              <button
                type="button"
                onClick={() => removeFile(i)}
                className="text-white/60 hover:text-white flex-shrink-0 ml-0.5"
                aria-label={`Remove ${file.name}`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-2 items-end">
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
              if ((input.trim() || attachedFiles.length > 0) && !isLoading && !isStreaming) {
                handleSubmit(e as unknown as FormEvent);
              }
            }
          }}
          placeholder="Ask me about sleep..."
          disabled={isLoading || isStreaming}
          rows={1}
          onPaste={handlePaste}
          className="input-sky flex-1 px-5 py-3 rounded-2xl border border-empower-200 dark:border-empower-600 bg-white dark:bg-empower-700 text-empower-800 dark:text-empower-100 placeholder:text-empower-300 dark:placeholder:text-empower-500 disabled:bg-empower-50 dark:disabled:bg-empower-800 disabled:text-empower-300 dark:disabled:text-empower-600 transition-shadow resize-none overflow-hidden"
          style={{ maxHeight: '96px' }}
        />

        {/* File attach button */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading || isStreaming || attachedFiles.length >= 5}
          title={attachedFiles.length >= 5 ? 'Max 5 files' : 'Attach PDF or image'}
          className="p-3 rounded-full border border-empower-200 dark:border-empower-600 text-empower-500 dark:text-empower-400 hover:bg-empower-50 dark:hover:bg-empower-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex-shrink-0"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
          </svg>
        </button>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/gif,application/pdf"
          multiple
          className="hidden"
          onChange={handleFileSelect}
        />

        <button
          type="submit"
          disabled={(!input.trim() && attachedFiles.length === 0) || isLoading || isStreaming}
          className="px-6 py-3 bg-empower-500 dark:bg-empower-600 text-white rounded-full font-medium hover:bg-empower-600 dark:hover:bg-empower-500 disabled:bg-empower-200 dark:disabled:bg-empower-700 disabled:cursor-not-allowed transition-colors shadow-sm flex-shrink-0"
        >
          Send
        </button>
      </div>

      <p className="text-xs text-empower-400 dark:text-empower-500 text-center mt-2">
        Educational information only. Not medical advice. AI can make mistakes — verify with a healthcare professional.{' '}
        By using, you accept{' '}
        <a href="https://www.empowersleep.com/terms-of-use" target="_blank" rel="noopener noreferrer" className="underline hover:text-empower-600 dark:hover:text-empower-300">Terms</a>
        {' & '}
        <a href="https://www.empowersleep.com/privacy-policy" target="_blank" rel="noopener noreferrer" className="underline hover:text-empower-600 dark:hover:text-empower-300">Privacy</a>.
      </p>
    </form>
  );

  /* ── HERO (no messages yet) ── */
  if (messages.length === 0) {
    return (
      <div className="relative flex flex-col h-screen overflow-hidden
        bg-[radial-gradient(ellipse_at_bottom,_#1a2f3f_0%,_#0d1a24_60%,_#060e14_100%)]
        dark:bg-[radial-gradient(ellipse_at_bottom,_#1a2f3f_0%,_#0d1a24_60%,_#060e14_100%)]
        [.light_&]:bg-none">

        {/* Light mode sunrise gradient */}
        <div className="absolute inset-0 dark:hidden
          bg-[radial-gradient(ellipse_at_bottom,_#fde68a_0%,_#fca5a5_30%,_#c4b5d4_60%,_#bfdbf7_100%)]" />

        {/* Stars (dark mode only) */}
        <div className="hidden dark:block">
          <StarField />
        </div>

        {/* Minimal top bar */}
        <div className="relative flex justify-end px-4 py-3 z-10">
          {themeToggleButton}
        </div>

        {/* Hero */}
        <main className="relative flex-1 flex flex-col items-center justify-center px-4 animate-fade-in z-10">
          <a
            href="https://www.empowersleep.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-80 transition-opacity mb-6"
          >
            <EmpowerLogo className="h-20 w-20 text-white dark:text-empower-200 drop-shadow-lg" />
          </a>
          <h1 className="text-4xl font-heading font-semibold text-white mb-3 text-center drop-shadow">
            EmpowerSleep
          </h1>
          <p className="text-white/70 text-center max-w-md mb-10">
            Ask me anything about sleep. I&apos;ll provide educational information grounded in expert content.
          </p>

          {/* Input — centered in hero */}
          <div className="w-full max-w-2xl mb-8">
            {inputForm}
          </div>

          {/* Sample question cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-2xl">
            {sampleQuestions.map((question) => (
              <button
                key={question}
                onClick={() => handleDemoClick(question)}
                className="text-left px-4 py-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl text-sm text-white/80 hover:bg-white/20 hover:text-white transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        </main>
      </div>
    );
  }

  /* ── CHAT (messages exist) ── */
  return (
    <div className="relative flex flex-col h-screen overflow-hidden">
      {/* Same background as hero */}
      <div className="absolute inset-0 dark:hidden
        bg-[radial-gradient(ellipse_at_bottom,_#fde68a_0%,_#fca5a5_30%,_#c4b5d4_60%,_#bfdbf7_100%)]" />
      <div className="absolute inset-0 hidden dark:block
        bg-[radial-gradient(ellipse_at_bottom,_#1a2f3f_0%,_#0d1a24_60%,_#060e14_100%)]" />
      <div className="hidden dark:block">
        <StarField />
      </div>

      {/* Minimal header — frosted glass */}
      <header className="relative z-10 bg-white/20 dark:bg-black/30 backdrop-blur-md border-b border-white/20 dark:border-white/10 px-4 py-3 flex items-center justify-between">
        <a
          href="https://www.empowersleep.com/"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 hover:opacity-80 transition-opacity"
        >
          <EmpowerLogo className="h-8 w-8 text-white" />
          <span className="font-heading font-semibold text-white">EmpowerSleep</span>
        </a>
        <div className="flex items-center gap-2">
          <button
            onClick={handleNewConversation}
            className="text-sm text-white/70 hover:text-white font-medium px-3 py-1.5 rounded-lg hover:bg-white/10 transition-colors"
          >
            New Chat
          </button>
          {themeToggleButton}
        </div>
      </header>

      {/* Chat area */}
      <main ref={chatContainerRef} onScroll={handleChatScroll} onWheel={handleWheel} className="relative z-10 flex-1 overflow-y-auto chat-scrollbar px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Messages */}
          {messages.map((message, index) => (
            <ChatMessage
              key={index}
              message={message}
              streaming={isStreaming && index === messages.length - 1 && message.role === 'assistant'}
              onRegenerate={message.role === 'assistant' ? () => handleRegenerate(index) : undefined}
              onFeedback={message.role === 'assistant' ? (rating) => handleFeedback(index, rating) : undefined}
            />
          ))}

          {/* Follow-up suggestions */}
          {suggestions.length > 0 && !isLoading && !isStreaming && (
            <div className="flex flex-wrap gap-2 animate-fade-in">
              {suggestions.map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    setSuggestions([]);
                    setInput(q);
                    trackEvent('suggestion_clicked', { suggestion: q });
                    setTimeout(() => {
                      if (inputRef.current) {
                        inputRef.current.focus();
                        inputRef.current.style.height = 'auto';
                        inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 96) + 'px';
                      }
                    }, 0);
                  }}
                  className="px-4 py-2 text-sm bg-white/20 dark:bg-white/10 backdrop-blur-sm border border-white/30 dark:border-white/20 rounded-full text-empower-800 dark:text-empower-100 hover:bg-white/30 dark:hover:bg-white/20 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {/* Loading indicator */}
          {isLoading && <SleepLoader />}

          {/* Error message */}
          {error && (
            <div className="flex justify-center animate-fade-in">
              <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-xl px-4 py-3 text-sm text-red-600 dark:text-red-400">
                {error}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input area */}
      <footer className="relative z-10 bg-white/20 dark:bg-black/30 backdrop-blur-md border-t border-white/20 dark:border-white/10 px-4 py-4">
        <div className="max-w-4xl mx-auto">
          {inputForm}
        </div>
      </footer>
    </div>
  );
}
