/**
 * lib/sleepThoughts.ts
 *
 * A collection of calm, non-medical sleep-related thoughts to display
 * while the AI is generating a response. These are NOT presented as facts
 * or medical information — just gentle, brand-appropriate musings.
 *
 * Guidelines for adding new thoughts:
 * - Keep them short (under 60 characters ideal)
 * - Avoid statistics, percentages, or specific claims
 * - No disease names or medical terminology
 * - Frame as observations, not prescriptions
 * - Maintain a calm, slightly warm tone
 */

export const SLEEP_THOUGHTS: string[] = [
  // Gentle observations about sleep
  "Your brain does a lot of quiet work while you rest.",
  "Sleep is when your mind tidies up.",
  "Rest isn't doing nothing — it's doing something different.",
  "Your body has its own rhythm. It's worth listening to.",
  "Even small improvements in rest can make a difference.",

  // Encouragement without prescription
  "Consistency tends to matter more than perfection.",
  "Good sleep habits are built one night at a time.",
  "Progress isn't always linear, and that's okay.",
  "Small changes can add up over time.",
  "There's no single \"right\" way to sleep well.",

  // Calm, reflective thoughts
  "The night is quieter than we often give it credit for.",
  "Sometimes the best thing to do is nothing at all.",
  "Rest is a skill, not just a state.",
  "Your sleep story is unique to you.",
  "Winding down is part of the process, not wasted time.",

  // Light and slightly playful
  "Even your brain appreciates a good night off.",
  "Pillows: underrated technology since forever.",
  "The snooze button isn't always the enemy.",
  "Counting sheep is optional.",
  "Dreams are your brain's way of being creative.",

  // Warmth and reassurance
  "You're taking a step by being curious about sleep.",
  "Better rest starts with better understanding.",
  "Asking questions is the first step.",
  "You're in the right place.",
];

/**
 * Get a random sleep thought.
 * Used when starting a new loading state.
 */
export function getRandomSleepThought(): string {
  const index = Math.floor(Math.random() * SLEEP_THOUGHTS.length);
  return SLEEP_THOUGHTS[index];
}
