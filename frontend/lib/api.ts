/**
 * lib/api.ts
 *
 * API client for communicating with the FastAPI backend.
 * All API calls go through this module for consistency.
 */

// Base URL from environment variable, fallback to localhost for development
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

/**
 * Source citation from the RAG system
 */
export interface Source {
  source_type: 'textbook' | 'blog' | 'web' | 'notion';
  title: string;
  chapter?: string;
  page_start?: number;
  page_end?: number;
  url?: string;
  snippet?: string;
}

/**
 * Chat response from the API
 */
export interface ChatResponse {
  answer: string;
  sources: Source[];
}

/**
 * File attachment on a user message
 */
export interface MessageAttachment {
  type: 'image' | 'pdf';
  name: string;
  url?: string; // object URL for inline image preview
}

/**
 * Message in conversation history
 */
export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  attachments?: MessageAttachment[];
}

/**
 * Send a chat message to the API and get a response.
 *
 * @param message - The user's message
 * @param history - Optional conversation history for context
 * @param signal - Optional AbortSignal to cancel the request
 * @returns Promise<ChatResponse> - The API response with answer and sources
 */
export async function sendMessage(
  message: string,
  history?: Message[],
  signal?: AbortSignal
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      history: history?.map(m => ({
        role: m.role,
        content: m.content,
      })),
    }),
    signal, // Pass abort signal to fetch
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Callbacks for the streaming chat API.
 */
export interface StreamCallbacks {
  onToken: (text: string) => void;
  onSources: (sources: Source[]) => void;
  onDone: () => void;
  onError: (err: Error) => void;
}

/**
 * Send a chat message to the streaming API endpoint.
 *
 * Reads the SSE response body and dispatches events via callbacks.
 * Sources arrive before the first token; [DONE] signals end of stream.
 *
 * @param message   - The user's message
 * @param history   - Optional conversation history for context
 * @param signal    - Optional AbortSignal to cancel the stream
 * @param callbacks - Event handlers for tokens, sources, completion, and errors
 */
export async function streamMessage(
  message: string,
  history?: Message[],
  signal?: AbortSignal,
  callbacks?: StreamCallbacks,
  files?: File[],
): Promise<void> {
  // Backend expects multipart/form-data (supports optional file attachments)
  const formData = new FormData();
  formData.append('message', message);
  formData.append(
    'history',
    JSON.stringify(history?.map((m) => ({ role: m.role, content: m.content })) ?? []),
  );
  if (files?.length) {
    files.forEach((f) => formData.append('files', f));
  }

  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    body: formData, // browser sets Content-Type + multipart boundary automatically
    signal,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() ?? '';

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;

        const dataStr = line.slice(5).trim();
        if (dataStr === '[DONE]') {
          callbacks?.onDone();
          return;
        }

        try {
          const event = JSON.parse(dataStr);
          if (event.type === 'token') {
            callbacks?.onToken(event.text);
          } else if (event.type === 'sources') {
            callbacks?.onSources(event.sources);
          } else if (event.type === 'error') {
            callbacks?.onError(new Error(event.message));
            return;
          }
        } catch {
          // ignore malformed SSE lines
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Submit thumbs up/down feedback for an AI response.
 *
 * @param sessionId    - Anonymous session identifier
 * @param userQuestion - The user's question
 * @param aiResponse   - The AI's response
 * @param rating       - 1 for thumbs up, -1 for thumbs down
 */
export async function submitFeedback(
  sessionId: string,
  userQuestion: string,
  aiResponse: string,
  rating: 1 | -1,
): Promise<void> {
  await fetch(`${API_BASE_URL}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      user_question: userQuestion,
      ai_response: aiResponse,
      rating,
    }),
  });
}

/**
 * Fetch 3 follow-up question suggestions after an AI response.
 */
export async function getSuggestions(
  message: string,
  response: string,
  history?: Message[],
  signal?: AbortSignal,
): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/suggestions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        response,
        history: history?.map((m) => ({ role: m.role, content: m.content })) ?? [],
      }),
      signal,
    });
    const data = await res.json();
    return data.suggestions ?? [];
  } catch {
    return [];
  }
}

/**
 * Format a source for display.
 *
 * Textbook: 📖 **Title** – Chapter X (pp. Y–Z)
 * Blog: [Title](url)
 *
 * @param source - The source to format
 * @returns Formatted string for display
 */
export function formatSourceDisplay(source: Source): {
  icon: string;
  text: string;
  url?: string;
} {
  if (source.source_type === 'textbook') {
    // Format: 📖 **Sleep and Health** – Chapter X (pp. Y–Z)
    let text = source.title;

    if (source.chapter) {
      text += ` – ${source.chapter}`;
    }

    if (source.page_start !== undefined) {
      if (source.page_end && source.page_end !== source.page_start) {
        text += ` (pp. ${source.page_start}–${source.page_end})`;
      } else {
        text += ` (p. ${source.page_start})`;
      }
    }

    return { icon: '📖', text };
  } else {
    // Blog/web source - return as link
    return {
      icon: '📝',
      text: source.title,
      url: source.url,
    };
  }
}
