'use client';

import { useMemo } from 'react';
import { getRandomSleepThought } from '@/lib/sleepThoughts';
import { EmpowerLogo } from './EmpowerLogo';

/**
 * SleepLoader component displays a branded, calming loading state
 * while the AI generates a response.
 *
 * Features:
 * - Animated logo with breathing top/bottom pieces + blue glow
 * - Animated floating "zzz" indicator
 * - Random sleep-themed thought that stays stable during loading
 *
 * The thought is selected once when the component mounts and stays
 * stable until unmounted (i.e., when loading completes).
 */
export function SleepLoader() {
  // Select a random thought once when component mounts.
  // useMemo with empty deps ensures it doesn't change during the loading period.
  const sleepThought = useMemo(() => getRandomSleepThought(), []);

  return (
    <div className="flex justify-start animate-fade-in">
      <div className="bg-white dark:bg-empower-800 border border-empower-100 dark:border-empower-700 rounded-2xl rounded-bl-md shadow-sm px-5 py-4 max-w-md">
        {/* Animated logo + zzz */}
        <div className="flex items-center gap-3 mb-2">
          <EmpowerLogo className="h-10 w-10 text-empower-500 dark:text-empower-400" animate />
          <div className="flex items-end gap-1 mt-2">
            <span className="text-empower-400 dark:text-empower-500 text-lg zzz-float zzz-1">z</span>
            <span className="text-empower-300 dark:text-empower-600 text-base zzz-float zzz-2">z</span>
            <span className="text-empower-200 dark:text-empower-700 text-sm zzz-float zzz-3">z</span>
          </div>
        </div>

        {/* Sleep thought with subtle label */}
        <div className="text-empower-500 dark:text-empower-400 text-sm">
          <span className="text-empower-400 dark:text-empower-500 text-xs uppercase tracking-wide block mb-1">
            While you wait…
          </span>
          <span className="italic">{sleepThought}</span>
        </div>
      </div>
    </div>
  );
}
