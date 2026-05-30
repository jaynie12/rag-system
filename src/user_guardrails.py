from __future__ import annotations

from langchain.agents.middleware import PIIDetectionError, PIIMiddleware

_BUILTIN_PII_TYPES = ("email", "credit_card", "ip", "mac_address", "url")

_API_KEY_PATTERN = (
    r"(?:sk-[a-zA-Z0-9]{20,}"
    r"|ghp_[a-zA-Z0-9]{36,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[a-zA-Z0-9-]{10,}"
    r"|OPENAI_API_KEY\s*=\s*\S+)"
)

_PHONE_PATTERN = (
    r"(?:\+\d{1,3}(?:[\s-]?\d){6,14}"
    r"|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)"
)

_UK_POSTCODE_PATTERN = r"(?i)\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"

_US_ADDRESS_PATTERN = (
    r"(?:\b\d{1,5}\s+\w+(?:\s+\w+)*\s+"
    r"(?:St|Street|Rd|Road|Ave|Avenue|Blvd|Lane|Ln|Drive|Dr)\b"
    r"|\b[A-Za-z]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b)"
)

_NATIONAL_ID_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"


def build_pii_middleware_rules() -> list[PIIMiddleware]:
    """Return PIIMiddleware rules that block structured PII in user questions."""
    block = {"strategy": "block", "apply_to_input": True, "apply_to_output": False}

    rules: list[PIIMiddleware] = [
        PIIMiddleware(pii_type, **block) for pii_type in _BUILTIN_PII_TYPES
    ]
    rules.extend(
        [
            PIIMiddleware("api_key", detector=_API_KEY_PATTERN, **block),
            PIIMiddleware("phone", detector=_PHONE_PATTERN, **block),
            PIIMiddleware(
                "uk_postcode",
                detector=_UK_POSTCODE_PATTERN,
                **block,
            ),
            PIIMiddleware(
                "us_address",
                detector=_US_ADDRESS_PATTERN,
                **block,
            ),
            PIIMiddleware(
                "national_id",
                detector=_NATIONAL_ID_PATTERN,
                **block,
            ),
        ]
    )
    return rules


def format_pii_rejection_message(error: PIIDetectionError) -> str:
    """Return a safe CLI message that does not echo matched sensitive text."""
    return (
        f"Question rejected: detected {error.pii_type}. "
        "Remove personal or sensitive data and try again."
    )


def validate_user_question(text: str, *, enabled: bool = True) -> None:
    """
    Scan user question text for structured PII and block before RAG processing.

    Raises:
        PIIDetectionError: When any configured rule detects PII (strategy=block).
    """
    if not enabled:
        return

    question = text.strip()
    if not question:
        return

    for rule in build_pii_middleware_rules():
        rule._process_content(question)
