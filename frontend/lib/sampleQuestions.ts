/**
 * lib/sampleQuestions.ts
 *
 * Curated list of frequently asked sleep questions.
 * These align with topics covered in the EmpowerSleep knowledge base
 * (blog articles + textbook content).
 *
 * 3 questions are randomly selected on each page load.
 */

export const SAMPLE_QUESTIONS: string[] = [
  // Sleep basics
  "What is sleep hygiene?",
  "How much sleep do I actually need?",
  "What happens to my body during sleep?",
  "What is REM sleep and why is it important?",

  // Common issues
  "Why do I wake up in the middle of the night?",
  "What causes insomnia?",
  "Why do I feel tired even after sleeping 8 hours?",
  "What should I do if I can't fall asleep?",

  // Lifestyle factors
  "How does caffeine affect sleep?",
  "Does alcohol help or hurt sleep?",
  "How does exercise affect my sleep?",
  "Should I nap during the day?",

  // Environment & habits
  "What's the ideal bedroom temperature for sleep?",
  "How does screen time affect sleep?",
  "What is a good bedtime routine?",
  "Does eating before bed affect sleep quality?",

  // Specific concerns
  "Why do I have trouble sleeping when stressed?",
  "What causes vivid dreams?",
  "Is it bad to hit the snooze button?",
  "How do I reset my sleep schedule?",

  // Health connections
  "How does sleep affect mental health?",
  "Can poor sleep affect my immune system?",
  "Why is sleep important for memory?",
];

/**
 * Get a random selection of sample questions.
 *
 * @param count - Number of questions to select (default: 3)
 * @returns Array of randomly selected questions
 */
export function getRandomQuestions(count: number = 3): string[] {
  // Shuffle using Fisher-Yates algorithm
  const shuffled = [...SAMPLE_QUESTIONS];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  return shuffled.slice(0, count);
}
