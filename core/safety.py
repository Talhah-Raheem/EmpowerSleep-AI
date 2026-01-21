"""
core/safety.py - Safety and Triage System
==========================================

This module implements a simple keyword-based triage system to detect
when users may need immediate help beyond what an educational chatbot
can provide.

IMPORTANT: This is a DEMO implementation using keyword matching.
A production system would use:
- ML-based intent classification
- More comprehensive phrase lists
- Integration with crisis hotline APIs
- Human review escalation paths

Triage Levels:
- CRISIS: Immediate danger to self - provide crisis resources
- URGENT: Medical concern requiring professional attention
- None: Safe to proceed with educational response
"""

from typing import Optional
from dataclasses import dataclass

# Import keyword lists from config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CRISIS_KEYWORDS, URGENT_KEYWORDS


@dataclass
class TriageResult:
    """
    Structured result from triage check.

    Attributes:
        level: "CRISIS" or "URGENT"
        message: User-facing message with resources/guidance
        matched_phrase: The phrase that triggered the triage (for logging)
    """
    level: str
    message: str
    matched_phrase: str


# =============================================================================
# RESPONSE MESSAGES
# =============================================================================

CRISIS_MESSAGE = """
🚨 **I'm concerned about what you've shared.**

If you're having thoughts of suicide or self-harm, please reach out for support:

**National Suicide Prevention Lifeline: 988** (call or text, 24/7)
**Crisis Text Line: Text HOME to 741741**
**International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/**

You don't have to face this alone. These services are free, confidential, and available 24/7.

I'm an educational chatbot and not equipped to provide crisis support. Please reach out to one of these resources or go to your nearest emergency room.
""".strip()


URGENT_MESSAGE = """
⚠️ **What you're describing sounds like it needs professional medical attention.**

I'm an educational tool and cannot provide medical advice. Please consider:

- **If this is an emergency**: Call 911 or go to your nearest emergency room
- **For urgent care**: Contact your doctor or visit an urgent care clinic
- **For sleep disorders**: Ask your doctor for a referral to a sleep specialist

Your health and safety are the priority. A healthcare professional can properly evaluate your symptoms and provide appropriate care.
""".strip()


# =============================================================================
# MAIN TRIAGE FUNCTION
# =============================================================================

def check_triage(user_text: str) -> Optional[TriageResult]:
    """
    Check user input for crisis or urgent medical indicators.

    This function scans the user's text for keywords/phrases that suggest
    they may need professional help rather than educational content.

    Args:
        user_text: The raw text input from the user

    Returns:
        TriageResult if escalation detected, None if safe to proceed

    Example:
        >>> result = check_triage("I can't sleep and I'm having thoughts of suicide")
        >>> result.level
        'CRISIS'
        >>> result = check_triage("What is sleep hygiene?")
        >>> result is None
        True
    """
    # Normalize text for matching (lowercase, extra spaces removed)
    normalized = user_text.lower().strip()

    # Check CRISIS keywords first (highest priority)
    for phrase in CRISIS_KEYWORDS:
        if phrase.lower() in normalized:
            return TriageResult(
                level="CRISIS",
                message=CRISIS_MESSAGE,
                matched_phrase=phrase
            )

    # Check URGENT keywords
    for phrase in URGENT_KEYWORDS:
        if phrase.lower() in normalized:
            return TriageResult(
                level="URGENT",
                message=URGENT_MESSAGE,
                matched_phrase=phrase
            )

    # No escalation needed
    return None


def get_standard_disclaimer() -> str:
    """
    Return the standard educational disclaimer to prepend to responses.

    This should be shown with every educational response to set
    appropriate expectations.
    """
    return (
        "ℹ️ *I'm an educational tool, not a medical professional. "
        "This information is for general learning only and should not "
        "replace advice from a qualified healthcare provider.*"
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_safe_to_respond(user_text: str) -> bool:
    """
    Quick check if it's safe to generate an educational response.

    Args:
        user_text: The user's input

    Returns:
        True if no triage escalation needed, False otherwise
    """
    return check_triage(user_text) is None


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    # Quick demo of the triage system
    test_inputs = [
        "What is sleep hygiene?",  # Safe
        "I can't sleep and I'm thinking about suicide",  # CRISIS
        "I haven't slept in days and I'm seeing things",  # URGENT
        "How does caffeine affect sleep?",  # Safe
        "I want to end my life",  # CRISIS
        "I took too much of my sleep medication",  # URGENT
    ]

    print("=" * 60)
    print("TRIAGE SYSTEM DEMO")
    print("=" * 60)

    for text in test_inputs:
        result = check_triage(text)
        if result:
            print(f"\n❌ INPUT: {text}")
            print(f"   LEVEL: {result.level}")
            print(f"   MATCHED: '{result.matched_phrase}'")
        else:
            print(f"\n✅ INPUT: {text}")
            print("   LEVEL: Safe to proceed")
