import pytest
from langchain.agents.middleware import PIIDetectionError

from src.user_guardrails import validate_user_question


@pytest.mark.parametrize(
    "question",
    [
        "Who is Mr. Darcy?",
        "What happens at the ball?",
        "Who is Elizabeth?",
    ],
)
def test_validate_user_question_allows_book_questions(question: str) -> None:
    validate_user_question(question)


@pytest.mark.parametrize(
    ("question", "expected_pii_type"),
    [
        ("My email is test@example.com", "email"),
        ("Card 4111 1111 1111 1111 please", "credit_card"),
        ("Key sk-" + "a" * 32, "api_key"),
        ("Call me at +44 7700 900123", "phone"),
        ("Postcode SW1A 1AA", "uk_postcode"),
        ("Address 123 Main St, Springfield, IL 62701", "us_address"),
        ("SSN 123-45-6789", "national_id"),
    ],
)
def test_validate_user_question_blocks_structured_pii(
    question: str,
    expected_pii_type: str,
) -> None:
    with pytest.raises(PIIDetectionError) as exc_info:
        validate_user_question(question)

    assert exc_info.value.pii_type == expected_pii_type


def test_validate_user_question_skipped_when_disabled() -> None:
    validate_user_question("test@example.com", enabled=False)
