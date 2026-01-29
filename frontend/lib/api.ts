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
  source_type: 'textbook' | 'blog' | 'web';
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
 * Message in conversation history
 */
export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
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
 * Check the health status of the API.
 *
 * @returns Promise<boolean> - True if API is healthy
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) return false;
    const data = await response.json();
    return data.status === 'healthy';
  } catch {
    return false;
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
      icon: '📄',
      text: source.title,
      url: source.url,
    };
  }
}
