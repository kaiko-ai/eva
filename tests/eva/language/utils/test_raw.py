"""Tests for raw text extraction utilities."""

import pytest

from eva.language.utils.text.raw import extract_raw

A_TO_Z_OPTIONS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Basic extraction patterns
        ("The answer is A", {"answer": "A"}),
        ("My choice is B", {"answer": "B"}),
        ("Answer: C", {"answer": "C"}),
        ("The correct answer is D.", {"answer": "D"}),
        ("A", {"answer": "A"}),
        ("B.", {"answer": "B"}),
        ("C) Answer text here", {"answer": "C"}),
        # Different answer formats
        ("I choose answer A", {"answer": "A"}),
        ("The right choice is B", {"answer": "B"}),
        ("Correct answer: C", {"answer": "C"}),
        ("My answer is D.", {"answer": "D"}),
        # Case variations
        ("answer is a", {"answer": "A"}),
        ("ANSWER IS B", {"answer": "B"}),
        ("Answer Is C", {"answer": "C"}),
        # With punctuation
        ("Answer: A.", {"answer": "A"}),
        ("Choice: B!", {"answer": "B"}),
        ("Answer is C:", {"answer": "C"}),
        # Whitespace variations
        ("Answer:   A  ", {"answer": "A"}),
        ("Answer :\n B", {"answer": "B"}),
        ("Answer\t:\tC", {"answer": "C"}),
        # Longer responses (tests tail extraction)
        (
            "This is a very long response with lots of text before the actual answer. " * 5
            + "The final answer is Z",
            {"answer": "Z"},
        ),
    ],
)
def test_extract_raw_basic_patterns(text: str, expected: dict) -> None:
    """Should extract answers from various basic patterns."""
    result = extract_raw(text, A_TO_Z_OPTIONS)
    assert result == expected


@pytest.mark.parametrize(
    ("text", "options", "expected"),
    [
        # With specific single-character options provided
        ("The answer is A", ["A", "B", "C"], {"answer": "A"}),
        ("I choose B", ["A", "B", "C"], {"answer": "B"}),
        ("Answer: C", ["A", "B", "C"], {"answer": "C"}),
        # With numeric single-character options
        ("The answer is 1", ["1", "2", "3"], {"answer": "1"}),
        ("I choose 2", ["1", "2", "3"], {"answer": "2"}),
        # Option not in provided list should not match
        ("The answer is Z", ["A", "B", "C"], None),
        ("I choose X", ["A", "B", "C"], None),
        # Multi-character options should work
        ("The answer is Yes", ["Yes", "No", "Maybe"], {"answer": "YES"}),
        ("I choose No", ["Yes", "No", "Maybe"], {"answer": "NO"}),
        ("Answer: Maybe", ["Yes", "No", "Maybe"], {"answer": "MAYBE"}),
        # Mixed single and multi-character should work as multi-character
        ("The answer is True", ["True", "False", "A"], {"answer": "TRUE"}),
        ("I choose A", ["True", "False", "A"], {"answer": "A"}),
    ],
)
def test_extract_raw_with_options(text: str, options: list[str], expected: dict | None) -> None:
    """Should extract answers based on provided options list."""
    result = extract_raw(text, options)
    assert result == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Without options, should default to A-Z
        ("The answer is A", {"answer": "A"}),
        ("I choose Z", {"answer": "Z"}),
        ("Answer: M", {"answer": "M"}),
        # Numbers should not match without explicit options
        ("The answer is 1", None),
        ("I choose 42", None),
        # Words should not match without explicit options
        ("The answer is Yes", None),
        ("I choose Blue", None),
    ],
)
def test_extract_raw_default_options(text: str, expected: dict | None) -> None:
    """Should default to A-Z options when none provided."""
    result = extract_raw(text, A_TO_Z_OPTIONS)
    assert result == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Multiple matches - should pick the last one
        ("First I thought A, but then B, finally C", {"answer": "C"}),
        ("Answer could be A or B or C", {"answer": "C"}),
        ("A B C D E F G", {"answer": "G"}),
        # Last match in different positions (matches first pattern that works)
        ("The answer A appears first, then B appears last", {"answer": "A"}),
        ("Options: A, B, C. I pick B", {"answer": "B"}),
    ],
)
def test_extract_raw_priority_last_occurrence(text: str, expected: dict) -> None:
    """Should prioritize the last match within each pattern tried."""
    result = extract_raw(text, A_TO_Z_OPTIONS)
    assert result == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Text with asterisks (should be removed)
        ("The answer is *A*", {"answer": "A"}),
        ("*Answer: B*", {"answer": "B"}),
        ("The answer is **C**", {"answer": "C"}),
        # Extra whitespace (should be normalized)
        ("Answer   :    A", {"answer": "A"}),
        ("The    answer    is    B", {"answer": "B"}),
        ("Answer:\n\n\nC", {"answer": "C"}),
        # Combination of cleaning
        ("**Answer  :   A**", {"answer": "A"}),
        ("*The    answer    is    B*", {"answer": "B"}),
    ],
)
def test_extract_raw_text_cleaning(text: str, expected: dict) -> None:
    """Should clean text by removing asterisks and normalizing whitespace."""
    result = extract_raw(text, A_TO_Z_OPTIONS)
    assert result == expected


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",  # Empty string
        "   ",  # Only whitespace
        None,  # None value
        123,  # Non-string type
        [],  # List
        {},  # Dict
    ],
)
def test_extract_raw_invalid_input(invalid_input) -> None:
    """Should return None for invalid inputs."""
    result = extract_raw(invalid_input, A_TO_Z_OPTIONS)
    assert result is None


@pytest.mark.parametrize(
    "text",
    [
        "This text has no answer pattern",
        "Just some random text here",
        "No valid options provided here",
        "The response is unclear",
        "I don't know",
        "answer",  # Just the word "answer" without a choice
        "The answer is",  # Incomplete pattern
    ],
)
def test_extract_raw_no_valid_answer(text: str) -> None:
    """Should return None when no valid answer pattern is found."""
    result = extract_raw(text, A_TO_Z_OPTIONS)
    assert result is None


def test_extract_raw_tail_extraction() -> None:
    """Should only examine the last 100 characters of long text."""
    # Create text longer than 100 chars with answer at the beginning (should be ignored)
    long_text_start = "The answer is A. " + "This is filler text. " * 10 + "More text here."
    result = extract_raw(long_text_start, A_TO_Z_OPTIONS)
    assert result is None  # A should not be found in the tail

    # Create text with answer in the tail (should be found)
    long_text_end = "This is filler text. " * 10 + "The final answer is B."
    result = extract_raw(long_text_end, A_TO_Z_OPTIONS)
    assert result == {"answer": "B"}


def test_extract_raw_case_insensitive_matching() -> None:
    """Should match answers case-insensitively but return uppercase."""
    test_cases = [
        ("answer: a", {"answer": "A"}),
        ("ANSWER: b", {"answer": "B"}),
        ("Answer: C", {"answer": "C"}),
        ("aNsWeR: d", {"answer": "D"}),
    ]

    for text, expected in test_cases:
        result = extract_raw(text, A_TO_Z_OPTIONS)
        assert result == expected


def test_extract_raw_pattern_variations() -> None:
    """Should match various answer pattern formats."""
    patterns = [
        # Standard patterns
        ("answer: A", {"answer": "A"}),
        ("choice: B", {"answer": "B"}),
        ("correct answer: C", {"answer": "C"}),
        ("right answer: D", {"answer": "D"}),
        ("right choice: E", {"answer": "E"}),
        ("correct choice: F", {"answer": "F"}),
        # Without colons
        ("answer A", {"answer": "A"}),
        ("choice B", {"answer": "B"}),
        ("correct answer C", {"answer": "C"}),
        # At end of line/text
        ("A", {"answer": "A"}),
        ("B.", {"answer": "B"}),
        ("C:", {"answer": "C"}),
        # With periods
        ("answer A.", {"answer": "A"}),
        ("choice B:", {"answer": "B"}),
        # With brackets
        ("A) Melanoma", {"answer": "A"}),
        ("B)", {"answer": "B"}),
        ("**C) Adenoma** is the correct answer", {"answer": "C"}),
    ]

    for text, expected in patterns:
        result = extract_raw(text, A_TO_Z_OPTIONS)
        assert result == expected, f"Failed for text: '{text}'"


def test_extract_raw_with_custom_single_char_options() -> None:
    """Should handle case properly with custom single-character options."""
    options = ["y", "n", "m"]

    test_cases = [
        ("answer: y", {"answer": "Y"}),
        ("answer: Y", {"answer": "Y"}),
        ("answer: n", {"answer": "N"}),
        ("answer: N", {"answer": "N"}),
        ("answer: m", {"answer": "M"}),
    ]

    for text, expected in test_cases:
        result = extract_raw(text, options)
        assert result == expected, f"Failed for text: '{text}'"


def test_extract_raw_with_multi_character_options() -> None:
    """Should handle multi-character options properly."""
    options = ["True", "False", "Unknown"]

    test_cases = [
        ("The answer is True", {"answer": "TRUE"}),
        ("I choose False", {"answer": "FALSE"}),
        ("Answer: Unknown", {"answer": "UNKNOWN"}),
        ("The correct answer is true", {"answer": "TRUE"}),  # Case insensitive
        ("Right choice: FALSE", {"answer": "FALSE"}),
        # Should not match partial words
        ("The answer is Truely", None),  # "True" in "Truely" shouldn't match
        ("I falsely believe", None),  # "False" in "falsely" shouldn't match
    ]

    for text, expected in test_cases:
        result = extract_raw(text, options)
        assert result == expected, f"Failed for text: '{text}'"


def test_extract_raw_mixed_option_lengths() -> None:
    """Should handle mixed single and multi-character options."""
    options = ["A", "Yes", "No", "B"]  # Mix of single and multi-character

    test_cases = [
        ("The answer is A", {"answer": "A"}),
        ("I choose Yes", {"answer": "YES"}),
        ("Answer: No", {"answer": "NO"}),
        ("The correct choice is B", {"answer": "B"}),
    ]

    for text, expected in test_cases:
        result = extract_raw(text, options)
        assert result == expected, f"Failed for text: '{text}'"


@pytest.mark.parametrize(
    ("text", "case_sensitive", "expected"),
    [
        # Case insensitive behavior
        ("The answer is Yes", False, {"answer": "YES"}),
        ("I choose yes", False, {"answer": "YES"}),
        ("Answer: YES", False, {"answer": "YES"}),
        ("The answer is No", False, {"answer": "NO"}),
        ("I choose no", False, {"answer": "NO"}),
        ("Answer: NO", False, {"answer": "NO"}),
        # Case sensitive behavior - should only match exact case
        ("The answer is Yes", True, {"answer": "Yes"}),  # Exact match
        ("I choose yes", True, None),  # Wrong case, should not match
        ("Answer: YES", True, None),  # Wrong case, should not match
        ("The answer is No", True, {"answer": "No"}),  # Exact match
        ("I choose no", True, None),  # Wrong case, should not match
        ("Answer: NO", True, None),  # Wrong case, should not match
        ("The answer is Maybe", True, {"answer": "Maybe"}),  # Exact match
        ("I choose maybe", True, None),  # Wrong case, should not match
    ],
)
def test_extract_raw_case_sensitive_behavior(
    text: str, case_sensitive: bool, expected: dict | None
) -> None:
    """Should handle case sensitivity properly."""
    options = ["Yes", "No", "Maybe"]
    result = extract_raw(text, options, case_sensitive=case_sensitive)
    assert result == expected


def test_extract_raw_default_case_insensitive() -> None:
    """Should default to case insensitive behavior."""
    options = ["Yes", "No", "Maybe"]
    test_cases = [
        ("The answer is Yes", {"answer": "YES"}),
        ("I choose yes", {"answer": "YES"}),
        ("Answer: YES", {"answer": "YES"}),
        ("The answer is No", {"answer": "NO"}),
        ("I choose no", {"answer": "NO"}),
        ("Answer: NO", {"answer": "NO"}),
    ]

    for text, expected in test_cases:
        result = extract_raw(text, options)  # No case_sensitive parameter
        assert result == expected, f"Default case insensitive failed for text: '{text}'"


@pytest.mark.parametrize(
    ("text", "case_sensitive", "expected"),
    [
        # Case insensitive behavior
        ("answer: A", False, {"answer": "A"}),
        ("answer: a", False, {"answer": "A"}),
        ("answer: b", False, {"answer": "B"}),
        ("answer: B", False, {"answer": "B"}),
        ("answer: c", False, {"answer": "C"}),
        ("answer: C", False, {"answer": "C"}),
        # Case sensitive behavior - should only match exact case
        ("answer: A", True, {"answer": "A"}),  # Exact match
        ("answer: a", True, None),  # Wrong case
        ("answer: b", True, {"answer": "b"}),  # Exact match
        ("answer: B", True, None),  # Wrong case
        ("answer: C", True, {"answer": "C"}),  # Exact match
        ("answer: c", True, None),  # Wrong case
    ],
)
def test_extract_raw_case_sensitive_single_char(
    text: str, case_sensitive: bool, expected: dict | None
) -> None:
    """Should handle case sensitivity with single character options."""
    options = ["A", "b", "C"]
    result = extract_raw(text, options, case_sensitive=case_sensitive)
    assert result == expected
