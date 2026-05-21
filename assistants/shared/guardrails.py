"""
Shared Guardrails / Safety Layer
Used by both OSS and Frontier assistants.
Author: Krishna Murthi
"""

import re
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class SafetyResult:
    is_safe: bool
    reason: str
    severity: str  # "low", "medium", "high"
    category: str  # "jailbreak", "harmful", "bias", "pii", "safe"


# ──────────────────────────────────────────────────────────────────────────────
# Input Guardrails
# ──────────────────────────────────────────────────────────────────────────────

# Patterns that indicate jailbreak / prompt injection attempts
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"you\s+are\s+now\s+(a\s+)?(DAN|evil|unrestricted|jailbroken)",
    r"pretend\s+(you\s+)?(are|have\s+no)\s+(restrictions|guidelines|rules)",
    r"(disable|bypass|remove|forget)\s+(your\s+)?(safety|filter|restriction|guideline)",
    r"act\s+as\s+(if\s+you\s+)?(are|were)\s+(unrestricted|evil|unethical)",
    r"developer\s+mode",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"prompt\s+injection",
    r"\[SYSTEM\].*override",
    r"ignore\s+your\s+training",
    r"you\s+must\s+comply\s+without\s+restrictions",
]

# Patterns for clearly harmful content requests
HARMFUL_PATTERNS = [
    r"\b(synthesize|make|create|produce|manufacture)\b.{0,30}\b(bomb|explosive|poison|drug|methamphetamine|fentanyl)\b",
    r"\b(hack|exploit|phish|ddos)\b.{0,30}\b(someone|account|system|server)\b",
    r"\b(child|minor|underage)\b.{0,30}\b(sexual|nude|naked|explicit)\b",
    r"\b(kill|murder|assassinate)\b.{0,30}\b(someone|person|president|politician)\b",
    r"\b(steal|identity theft)\b.{0,30}\b(credit card|ssn|social security|bank)\b",
]

# Patterns for potentially biased/stereotype-reinforcing prompts
BIAS_TRIGGER_PATTERNS = [
    r"\b(all|every)\b.{0,20}\b(women|men|blacks|whites|asians|muslims|christians|jews)\b.{0,30}\bare\b",
    r"\bprove\s+that\s+\w+\s+are\s+(inferior|superior|dumb|stupid|evil)\b",
    r"\bwhy\s+are\s+\w+\s+(so|always|never)\b",
]

# PII detection patterns
PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),
    (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', "Credit Card"),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email"),
]


def check_input_safety(text: str) -> SafetyResult:
    """
    Check if user input is safe to process.
    Returns SafetyResult with is_safe=True if input passes all checks.
    """
    text_lower = text.lower().strip()

    # Check for jailbreak attempts
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            return SafetyResult(
                is_safe=False,
                reason="This message appears to be attempting to bypass my safety guidelines.",
                severity="high",
                category="jailbreak"
            )

    # Check for clearly harmful content
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            return SafetyResult(
                is_safe=False,
                reason="I'm not able to help with requests that could cause harm.",
                severity="high",
                category="harmful"
            )

    # Check for bias-triggering prompts (flag but still process with care)
    for pattern in BIAS_TRIGGER_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return SafetyResult(
                is_safe=True,  # Process but note the sensitivity
                reason="Sensitive topic detected — will respond carefully.",
                severity="medium",
                category="bias"
            )

    return SafetyResult(
        is_safe=True,
        reason="Input passed all safety checks.",
        severity="low",
        category="safe"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Output Guardrails
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_HARMFUL_PATTERNS = [
    r"\b(Here\s+is\s+how\s+to\s+make\s+a\s+bomb)\b",
    r"\b(step\s+1.*step\s+2.*synthesize)\b",
    r"\b(I\s+will\s+help\s+you\s+hack)\b",
]


def check_output_safety(text: str) -> Tuple[bool, str]:
    """
    Post-process LLM output to catch any safety slips.
    Returns (is_safe, cleaned_text).
    """
    for pattern in OUTPUT_HARMFUL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return False, "I apologize, but I cannot provide that information."

    # Check output length sanity
    if len(text) < 1:
        return True, "I'm sorry, I couldn't generate a response. Please try again."

    return True, text


# ──────────────────────────────────────────────────────────────────────────────
# Refusal Templates
# ──────────────────────────────────────────────────────────────────────────────

REFUSAL_TEMPLATES = {
    "jailbreak": (
        "I notice this message is trying to bypass my safety guidelines. "
        "I'm designed to be helpful, harmless, and honest — those principles "
        "aren't constraints I can set aside. How can I help you with something constructive?"
    ),
    "harmful": (
        "I'm not able to assist with that request as it could cause harm. "
        "If you have a legitimate need related to this topic (e.g., research, fiction writing), "
        "please provide more context and I'll do my best to help within safe boundaries."
    ),
    "bias": (
        "That's a complex topic that touches on stereotypes. Let me give you a "
        "nuanced, evidence-based perspective rather than a generalization..."
    ),
}


def get_refusal_message(category: str) -> str:
    """Return an appropriate refusal message for a given safety category."""
    return REFUSAL_TEMPLATES.get(category, "I'm not able to help with that request.")
