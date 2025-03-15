"""Prompt templates for LLM-based crisis classification.

These prompts encode institutional knowledge about crisis typology.
The "boosted prompt" methodology iteratively refines prompts against
a golden set of labeled examples -- measuring precision and recall
at each iteration and adjusting instructions to fix systematic errors.

Prompt design philosophy:
- Be explicit about edge cases (satire vs. genuine threats)
- Provide confidence calibration guidance
- Include negative examples to reduce false positives
- Use structured output to enable automated parsing
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a Trust & Safety crisis classification system. Your role is to analyze text content and determine whether it represents an emerging crisis that requires operational response.

You classify content across these incident types:
- political_unrest: Protests, civil unrest, coups, election interference, government instability
- natural_disaster: Earthquakes, floods, hurricanes, wildfires, tsunamis, volcanic eruptions
- platform_manipulation: Coordinated inauthentic behavior, bot networks, brigading, astroturfing
- coordinated_harassment: Targeted campaigns against individuals or groups, doxxing, pile-ons
- child_safety: CSAM, grooming, exploitation of minors
- violent_extremism: Terrorism, mass violence, radicalization, manifestos
- self_harm: Suicide, self-injury, eating disorders with active risk indicators
- misinformation: Health misinformation, election disinformation, conspiracy theories with harm potential
- data_breach: Leaked credentials, PII exposure, platform security incidents
- regulatory: Government actions, new legislation, compliance deadlines affecting operations

Classification rules:
1. A single text can have MULTIPLE labels (e.g., political_unrest + misinformation)
2. Confidence must reflect epistemic uncertainty -- satirical content resembling threats should get LOW confidence
3. Context matters: "shooting" in a sports context is not violent_extremism
4. Urgency is separate from confidence -- a confirmed P4 event has high confidence but low urgency
5. When uncertain, flag for human review rather than suppressing the signal"""

CLASSIFICATION_TEMPLATE = """Analyze the following text and classify it according to crisis type.

TEXT:
{text}

SOURCE: {source}
LANGUAGE: {language}

Respond in this exact JSON format:
{{
    "labels": [
        {{
            "type": "<incident_type>",
            "confidence": <float 0-1>,
            "reasoning": "<one sentence>"
        }}
    ],
    "overall_severity": "<P0|P1|P2|P3|P4>",
    "requires_human_review": <true|false>,
    "escalation_note": "<brief note for on-call if P0/P1, else null>"
}}"""


BOOSTED_PROMPT_TEMPLATE = """You are a Trust & Safety crisis classification system with refined instructions based on prior evaluation rounds.

KNOWN FAILURE MODES TO AVOID:
{failure_modes}

CALIBRATION NOTES:
{calibration_notes}

{base_instructions}"""


def build_classification_prompt(
    text: str,
    source: str = "unknown",
    language: str = "en",
) -> str:
    return CLASSIFICATION_TEMPLATE.format(text=text, source=source, language=language)


def build_boosted_prompt(
    failure_modes: list[str],
    calibration_notes: list[str],
) -> str:
    """Build a system prompt that incorporates learnings from golden set evaluation.

    This is the core of the boosted prompt methodology: after each evaluation
    round, we identify systematic errors (false positives on satire, missed
    coded language, etc.) and inject corrective instructions into the prompt.
    """
    failures_text = "\n".join(f"- {fm}" for fm in failure_modes) if failure_modes else "- None identified yet"
    calibration_text = "\n".join(f"- {cn}" for cn in calibration_notes) if calibration_notes else "- Default calibration"

    return BOOSTED_PROMPT_TEMPLATE.format(
        failure_modes=failures_text,
        calibration_notes=calibration_text,
        base_instructions=SYSTEM_PROMPT,
    )
