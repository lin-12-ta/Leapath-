"""LangGraph orchestration for the session-aware Career Navigator."""
import difflib
import functools
import json
import re
import time
from typing import Literal, TypedDict

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .config import settings
from .rag import retrieve

MIN_MATCH_SCORE = 0.15
MAX_CRITIC_REVISIONS = 1
LLM_MAX_ATTEMPTS = 3          # chain-level attempts (malformed JSON, transient errors)
LLM_CLIENT_RETRIES = 2        # client-level retries (honours Retry-After on 429/5xx)
LLM_COOLDOWN_SECONDS = 30
LLM_COOLDOWN_MAX_SECONDS = 120
_llm_cooldown_until = 0.0

PROFILE_FIELDS = ("interests", "work_style", "skills_background", "values_constraints")
ENTRY_FIELDS = {"What they do", "Skills", "Who thrives here", "Work environment", "Outlook"}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
INTAKE_PROMPT = """You are the Intake agent in a career guidance system. Learn about the user through a natural, multi-turn conversation and build a structured profile. You do NOT recommend careers.

Coverage areas:
- interests: topics or activities the user enjoys or loses track of time in
- work_style: solo/team, structured/flexible, hands-on/desk-based, pace, people-facing/behind the scenes
- skills_background: education, experience, and strengths the user says they have
- values_constraints: income, stability, meaning, flexibility, location, schooling, hours, health, or other limits

Read the full conversation and existing profile. Ask exactly ONE warm, natural, open question per turn, based on what the user just said. Follow up on vague answers before changing areas. Never re-ask something already answered; never present a form or numbered list. Only include facts the user stated or clearly implied. Update each coverage rating to none, weak, or solid.

Set status ready only when interests AND work_style are solid, at least one of skills_background or values_constraints is solid, and the profile has concrete details that distinguish this user; OR when the user asks to see results, declines further questions, or the existing intake question count has reached 8. Otherwise set status ask and ask about the weakest coverage area. When ready, use a brief transition instead of a question. Do not handle crisis or unrelated support needs; set flags and leave that to Safety.

Return only the requested IntakeResult JSON."""

MATCHER_PROMPT = """You are the Career Matcher agent. Propose career directions based ONLY on the retrieved knowledge base entries. Do not rely on general career knowledge.

Recommend 2 to 3 careers, only using exact titles in the retrieved entries. Never recommend a title listed in excluded_titles. Do not state duties, skills, outlook, work environment, pay, or other facts absent from those entries; if the entries do not cover a question, say so. For each recommendation, link specific user statements (quote or closely paraphrase what the user actually said) to named entry fields (What they do, Skills, Who thrives here, Work environment, Outlook), cite the exact title and similarity score, and honestly state at least one possible mismatch. Never violate a hard user constraint without explicitly flagging the conflict. Suggest a small next step supported by the entry or a general exploration step. Encourage without guaranteeing outcomes.

If Critic feedback is supplied, correct the identified issues without introducing new unsupported claims. If the best score is below the supplied minimum, or the retrieved careers clearly do not fit, return no_strong_match=true, no recommendations, explain why, and identify what information or evidence is missing. Otherwise return 2 to 3 recommendations. Return only MatcherResult JSON."""

CRITIC_PROMPT = """You are the Critic and Safety agent. Do two separate jobs and report them separately. Never rewrite the recommendations.

JOB A — CONSISTENCY: Compare the full conversation, profile, matcher output, and retrieved entries. Identify contradictions with stated dislikes, hard constraints, refused options (profile.rejected_titles), or work style; ungrounded claims or titles not retrieved; misquoted user statements; and missed constraints. Each issue must have severity minor or major and evidence from both sides. Do not invent problems. If clean, verdict pass with no issues; otherwise verdict revise.

JOB B — SAFETY: Read the whole conversation (not only the latest message) for serious distress, crisis, self-harm, severe anxiety/depression/burnout/panic affecting daily functioning, abuse/unsafe situations, urgent housing/income/legal/immigration needs, specialist medical/disability needs, or a minor needing age-appropriate support. Ordinary career stress, feeling lost, or ordinary money worries are not enough by themselves. When unsure, choose note. Levels are none, note, redirect. For redirect, write a short warm message: acknowledge, do not diagnose, encourage an appropriate professional or trusted person, mention local emergency/crisis help if safety is at risk, and offer to return to career planning later.

Return only CriticResult JSON."""

SAFETY_SCREEN_PROMPT = """Screen the full conversation for needs other than career advice using the safety criteria: imminent self-harm/crisis, severe mental health symptoms affecting daily life, abuse/unsafe conditions, urgent housing/income/legal/immigration needs, specialist medical/disability support, or a minor needing age-appropriate guidance. Ordinary career stress or money worries alone are not a reason to redirect. Loss of interest in everything or describing oneself as worthless may merit a note when context is unclear. Choose none, note, or redirect. Do not diagnose. For redirect provide a short warm hand-off message. For note, acknowledge the concern briefly and pause career questions for this turn; ask one gentle question about immediate safety or trusted support when the message suggests emotional distress. Return only SafetyReview JSON."""

FOLLOWUP_PROMPT = """You are the Follow-up agent. The user has already seen career recommendations. Classify their latest message and act on it. You do NOT invent career facts.

Intents:
- reject: the user dislikes or rules out one or more of the previously recommended careers. Put the exact titles in rejected_titles (choose only from known_titles).
- refine: the user adds or changes a preference, skill, or constraint and wants updated suggestions. Put new items in profile_updates using only the keys interests, work_style, skills_background, values_constraints. Only include what the user actually said.
- explain: the user asks about a career or wants more detail. Write `answer` using ONLY the retrieved_entries; if they do not cover the question, say so plainly. Do not add facts from general knowledge.
- restart: the user wants to start over.
- other: anything else. Write a short, warm `answer` that asks what they would like to do next.

A message can both reject and refine (e.g. "not the first one, I want something more hands-on"); use intent reject and also fill profile_updates. Return only FollowupResult JSON."""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Coverage(BaseModel):
    interests: Literal["none", "weak", "solid"] = "none"
    work_style: Literal["none", "weak", "solid"] = "none"
    skills_background: Literal["none", "weak", "solid"] = "none"
    values_constraints: Literal["none", "weak", "solid"] = "none"


class Profile(BaseModel):
    interests: list[str] = Field(default_factory=list)
    work_style: list[str] = Field(default_factory=list)
    skills_background: list[str] = Field(default_factory=list)
    values_constraints: list[str] = Field(default_factory=list)
    rejected_titles: list[str] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    questions_asked: int = Field(default=0, ge=0)
    history_start_index: int = Field(default=0, ge=0)


class IntakeResult(BaseModel):
    status: Literal["ask", "ready"]
    message: str
    profile: Profile
    coverage: Coverage
    flags: list[str] = Field(default_factory=list)
    reason: str


class Recommendation(BaseModel):
    title: str
    similarity_score: float = Field(ge=-1, le=1)
    why_it_fits: str
    matched_user_statements: list[str] = Field(default_factory=list)
    matched_entry_fields: list[str] = Field(default_factory=list)
    possible_mismatch: str
    next_step: str


class MatcherResult(BaseModel):
    recommendations: list[Recommendation] = Field(default_factory=list)
    no_strong_match: bool = False
    gaps_or_questions: str = ""
    summary_for_user: str = ""


class ConsistencyIssue(BaseModel):
    severity: Literal["minor", "major"]
    type: Literal["contradiction", "ungrounded", "misquote", "missed_constraint"]
    detail: str
    evidence: str


class ConsistencyReview(BaseModel):
    verdict: Literal["pass", "revise"]
    issues: list[ConsistencyIssue] = Field(default_factory=list)


class SafetyReview(BaseModel):
    level: Literal["none", "note", "redirect"] = "none"
    signals: list[str] = Field(default_factory=list)
    rationale: str = "No concerning signal identified."
    message_to_user: str = ""


class CriticResult(BaseModel):
    consistency: ConsistencyReview
    safety: SafetyReview


class FollowupResult(BaseModel):
    intent: Literal["reject", "refine", "explain", "restart", "other"]
    rejected_titles: list[str] = Field(default_factory=list)
    profile_updates: dict[str, list[str]] = Field(default_factory=dict)
    answer: str = ""


class Flow(TypedDict, total=False):
    message: str
    profile: dict
    history: list[dict]
    history_offset: int
    stage: str
    retrieved: list[dict]
    matches: list[dict]
    matcher_result: dict
    critique: dict
    safety_review: dict
    intake_message: str
    intake_flags: list
    coverage: dict
    followup_answer: str
    followup_intent: str
    reply: str
    route: str
    status: str
    summary: str
    review_attempts: int


# ---------------------------------------------------------------------------
# Retrieval as a LangChain tool
# ---------------------------------------------------------------------------
@tool
def search_careers(query: str, k: int = 4) -> list[dict]:
    """Search the career knowledge base and return the top-k entries with similarity scores."""
    return retrieve(query, k)


# ---------------------------------------------------------------------------
# LLM plumbing (LangChain + retry/backoff/cooldown)
# ---------------------------------------------------------------------------
# Transient failures worth retrying at chain level. OutputParserException covers
# malformed or schema-invalid JSON, which a fresh sample usually fixes.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    OutputParserException,
)


def _status_of(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = (
        getattr(exc, "status_code", None)
        or getattr(response, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(exc, "status", None)
    )
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _safe_llm_error(exc: Exception) -> str:
    """Format provider diagnostics without exposing credentials or request headers."""
    detail = str(getattr(exc, "message", None) or exc)
    detail = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED_API_KEY]", detail)
    detail = re.sub(r"sk-or-v1-[0-9A-Za-z]{20,}", "[REDACTED_API_KEY]", detail)
    detail = re.sub(r"(?i)([?&]key=)[^&\s]+", r"\1[REDACTED]", detail)
    detail = re.sub(r"(?i)(api[_ -]?key\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", detail)
    detail = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+", r"\1[REDACTED]", detail)
    detail = " ".join(detail.split())[:400]
    return f"error_type={type(exc).__name__} status={_status_of(exc) or 'unknown'} detail={detail or 'no provider detail'}"


def _cooldown_seconds(exc: Exception) -> float:
    """Seconds to skip the provider after a rate limit or server error (0 = no cooldown)."""
    status = _status_of(exc)
    if status is None or not (status == 429 or status >= 500):
        return 0.0
    seconds = float(LLM_COOLDOWN_SECONDS)
    headers = getattr(getattr(exc, "response", None), "headers", None)
    retry_after = headers.get("retry-after") if headers else None
    try:
        if retry_after is not None:
            seconds = max(seconds, float(retry_after))
    except (TypeError, ValueError):
        pass
    return min(seconds, float(LLM_COOLDOWN_MAX_SECONDS))


@functools.lru_cache(maxsize=1)
def _chat_model() -> ChatOpenAI:
    # max_retries lets the OpenAI client retry 429/5xx and honour Retry-After itself.
    return ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        timeout=30,
        max_retries=LLM_CLIENT_RETRIES,
    )


def _llm(system: str, user: str, schema: type[BaseModel]) -> BaseModel | None:
    """LangChain chain (ChatOpenAI | PydanticOutputParser) with retry, backoff, and cooldown.

    Returns None on failure so each node can use its validated offline fallback.
    """
    global _llm_cooldown_until
    if not settings.openrouter_api_key:
        return None
    remaining = _llm_cooldown_until - time.monotonic()
    if remaining > 0:
        print(f"[LLM] provider_cooldown schema={schema.__name__} remaining_seconds={remaining:.0f} fallback=offline", flush=True)
        return None

    parser = PydanticOutputParser(pydantic_object=schema)
    chain = (
        _chat_model().bind(response_format={"type": "json_object"}) | parser
    ).with_retry(
        retry_if_exception_type=_RETRYABLE,
        wait_exponential_jitter=True,
        stop_after_attempt=LLM_MAX_ATTEMPTS,
    )
    messages = [
        SystemMessage(content=f"{system}\n\nReturn a JSON object matching this schema:\n{json.dumps(schema.model_json_schema())}"),
        HumanMessage(content=user),
    ]
    print(f"[LLM] start schema={schema.__name__}", flush=True)
    try:
        result = chain.invoke(messages)  # output is validated by the Pydantic parser
        print(f"[LLM] success schema={schema.__name__}", flush=True)
        return result
    except Exception as exc:
        print(f"[LLM] failed schema={schema.__name__} {_safe_llm_error(exc)} fallback=offline", flush=True)
        cooldown = _cooldown_seconds(exc)
        if cooldown:
            _llm_cooldown_until = time.monotonic() + cooldown
            print(f"[LLM] provider_cooldown_started seconds={cooldown:.0f}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------
def _profile_from_state(raw: dict | None) -> Profile:
    """Accept profiles stored by earlier app versions and map their legacy fields."""
    raw = dict(raw or {})
    raw.setdefault("skills_background", raw.get("strengths", []))
    raw.setdefault("values_constraints", raw.get("constraints", []))
    return Profile.model_validate(raw)


_EXTRACTION_GROUPS = {
    "interests": {
        "clay modelling": ("clay", "clay modelling", "clay modeling", "pottery", "ceramic", "sculpt", "wheel throwing", "glazing"),
        "visual design": ("design", "drawing", "illustrat", "graphic", "visual art"),
        "psychology": ("psychology", "human behavior", "human behaviour"),
        "data and analysis": ("data", "statistics", "analytics", "spreadsheet"),
        "software and technology": ("coding", "programming", "software", "technology"),
        "health and care": ("healthcare", "health", "nursing", "patient care"),
        "teaching and learning": ("teach", "education", "tutoring", "mentoring"),
        "writing and communication": ("writing", "journalism", "storytelling", "communication"),
        "environment and science": ("environment", "sustainability", "wildlife", "science"),
        "business and marketing": ("business", "marketing", "sales", "entrepreneur"),
    },
    "skills_background": {
        "creative": ("creative", "imaginative", "come up with ideas"),
        "analytical": ("analytical", "analyzing", "analysing", "analysis", "logic", "numbers"),
        "detail-oriented": ("detail", "precise", "patient", "careful"),
        "communication": ("communicate", "explaining", "explain", "writing"),
        "supporting others": ("helping people", "supporting people", "empathy", "empathetic"),
        "organizing": ("organize", "organise", "planning", "coordinating"),
        "teaching": ("teaching", "teach", "instructing", "mentoring"),
        "hands-on making": ("good with my hands", "making things", "crafting", "clay", "pottery", "sculpt"),
    },
    "work_style": {
        "collaborative": ("team", "collaborat", "working with people"),
        "independent": ("independent", "on my own", "by myself", "solo"),
        "hands-on": ("hands-on", "practical work", "in a studio", "making things"),
        "structured": ("structured", "routine", "clear instructions"),
        "flexible": ("flexible", "variety", "changing tasks"),
        "remote": ("remote", "from home"),
        "people-facing": ("people-facing", "with customers", "with clients"),
        "desk-based": ("desk-based", "at a desk"),
    },
    "values_constraints": {
        "income stability matters": ("steady income", "stable income", "reliable income", "salary"),
        "location matters": ("location", "near me", "where i live", "relocate"),
        "training time matters": ("degree", "training", "qualification", "time to study", "no more school"),
        "flexible schedule matters": ("flexible hours", "part-time", "caregiving", "schedule"),
        "meaningful work matters": ("meaningful", "make a difference", "purpose"),
    },
}


def _has_alias(lowered: str, alias: str) -> bool:
    # Word-boundary at the START only, so stems like "illustrat" or "collaborat" still
    # match, but "data" no longer matches inside "candidate" and "team" not in "steam".
    return re.search(r"(?<![a-z0-9])" + re.escape(alias), lowered) is not None


def _fallback_extract_profile(profile: Profile, text: str) -> Profile:
    """Small offline extractor; only adds signals explicitly present in the text."""
    lowered = text.lower()
    for field, labels in _EXTRACTION_GROUPS.items():
        current = getattr(profile, field)
        for label, aliases in labels.items():
            if any(_has_alias(lowered, alias) for alias in aliases) and label not in current:
                current.append(label)
    return profile


def _coverage_for(profile: Profile) -> Coverage:
    return Coverage(**{
        field: ("solid" if len(getattr(profile, field)) >= 1 else "none")
        for field in PROFILE_FIELDS
    })


def _asks_for_results(message: str) -> bool:
    text = message.lower()
    return any(term in text for term in ("show me results", "show my results", "see results", "give me recommendations", "recommend careers", "what careers suit", "no more questions", "stop asking", "i don't want more questions", "i do not want more questions"))


def _one_follow_up(profile: Profile, coverage: Coverage) -> str:
    weakest = min(PROFILE_FIELDS, key=lambda field: {"none": 0, "weak": 1, "solid": 2}[getattr(coverage, field)])
    questions = {
        "interests": "What kinds of activities or problems do you enjoy enough to keep coming back to?",
        "work_style": "What kind of work setting helps you do your best work?",
        "skills_background": "What experience, education, or strengths would you like your future work to build on?",
        "values_constraints": "What matters most in a job for you, or is there anything a job needs to accommodate?",
    }
    if weakest == "interests" and profile.interests:
        return "You mentioned " + ", ".join(profile.interests[:2]) + ". What do you most enjoy doing within that?"
    return questions[weakest]


def _user_texts(state: Flow) -> list[str]:
    texts = [m.get("content", "") for m in state.get("history", []) if m.get("role") == "user"]
    texts.append(state.get("message", ""))
    return texts


def _ground_statements(statements: list[str], user_texts: list[str], profile: Profile) -> list[str]:
    """Keep only statements traceable to what the user said or to the stored profile."""
    corpus = " ".join(user_texts).lower()
    originals = [item for field in PROFILE_FIELDS for item in getattr(profile, field)]
    items = [item.lower() for item in originals]
    kept = []
    for statement in statements:
        low = statement.lower().strip()
        if not low:
            continue
        if low in corpus or any(low in item or item in low for item in items) or difflib.get_close_matches(low, items, n=1, cutoff=0.7):
            kept.append(statement)
    return kept or originals


_LEVEL_RANK = {"none": 0, "note": 1, "redirect": 2}


def _merge_safety(earlier: SafetyReview, later: SafetyReview) -> SafetyReview:
    """Keep the more severe review; ties go to the later (more contextual) one."""
    return later if _LEVEL_RANK[later.level] >= _LEVEL_RANK[earlier.level] else earlier


def _has_unnegated_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    negation = re.compile(r"\b(?:not|never|no|don't|dont|do not|am not|aren't|isn't)\b(?:\W+\w+){0,3}\W*$", re.IGNORECASE)
    for phrase in phrases:
        start = 0
        while (index := text.find(phrase, start)) >= 0:
            prefix = text[max(0, index - 48):index]
            if not negation.search(prefix):
                return True
            start = index + len(phrase)
    return False


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def safety_screen(state: Flow) -> Flow:
    """Runs on EVERY turn. Then routes by session stage: support / pause / followup / intake."""
    conversation = state.get("history", []) + [{"role": "user", "content": state.get("message", "")}]
    context = json.dumps({"conversation": conversation, "stage": state.get("stage", "intake")})
    review = _llm(SAFETY_SCREEN_PROMPT, context, SafetyReview)
    # Deterministic overlay so an obvious distress signal cannot be lost to an LLM false negative.
    text = state.get("message", "").lower()
    redirect_terms = ("kill myself", "want to die", "suicidal", "suicide", "self harm", "self-harm", "hurt myself", "in immediate danger", "being abused", "about to lose housing", "homeless tonight", "no income", "immigration lawyer", "legal emergency")
    distress_terms = ("good for nothing", "worthless", "nothing interests me", "no interest in anything", "not interested in anything", "i don't enjoy anything", "i dont enjoy anything", "everything feels pointless", "hopeless", "severe anxiety", "panic attacks", "depressed every day", "burned out", "burnt out", "harassed", "unsafe at work")
    age_terms = ("i am 15", "i'm 15", "i am 16", "i'm 16")
    if _has_unnegated_phrase(text, redirect_terms):
        review = SafetyReview(level="redirect", signals=["The conversation contains a serious safety or urgent-needs signal."], rationale="This needs support beyond career guidance.", message_to_user="I’m sorry you’re facing this. Please reach out to someone you trust or an appropriate local professional now. If you may be in immediate danger or might hurt yourself, contact local emergency services or a crisis line. We can return to career planning later.")
    elif _has_unnegated_phrase(text, distress_terms) and (review is None or review.level == "none"):
        review = SafetyReview(level="note", signals=["The user described loss of interest or strongly negative feelings about themselves."], rationale="This may indicate emotional distress; acknowledge it without diagnosing or assuming immediate danger.", message_to_user="I’m sorry you’re feeling this way. Saying you feel like you’re good for nothing sounds painful. We can pause career questions for now. Are you safe right now, and is there someone you trust you could talk with?")
    elif _has_unnegated_phrase(text, age_terms) and (review is None or review.level == "none"):
        review = SafetyReview(level="note", signals=["The user may be a minor and may need age-appropriate guidance."], rationale="A brief age-appropriate check-in is appropriate.", message_to_user="Thanks for sharing that. I’ll keep the guidance appropriate to your age. Is there a trusted adult who can support you as you explore your options?")
    elif review is None:
        review = SafetyReview()
    validated = SafetyReview.model_validate(review.model_dump())

    update: Flow = {}
    if validated.level == "redirect":
        route = "support"
        update["stage"] = "support"
    elif validated.level == "note":
        route = "pause"
    else:
        # Session-state-driven dispatch: users who already have recommendations get the follow-up path.
        route = "followup" if state.get("stage") in ("recommendations", "review") else "intake"
    print(f"[FLOW] safety_screen -> {route} safety={validated.level} stage={state.get('stage', 'intake')}", flush=True)
    return {**update, "safety_review": validated.model_dump(), "route": route}


def intake(state: Flow) -> Flow:
    profile = _profile_from_state(state.get("profile"))
    history = state.get("history", [])
    conversation = history + [{"role": "user", "content": state.get("message", "")}]
    # The persisted count already includes the prior assistant question.
    question_count = profile.questions_asked
    existing = profile.model_copy(deep=True)
    existing.questions_asked = question_count
    context = json.dumps({"conversation": conversation, "existing_profile": existing.model_dump(), "intake_questions_asked": question_count}, ensure_ascii=False)
    result = _llm(INTAKE_PROMPT, context, IntakeResult)

    if result is None:
        updated = _fallback_extract_profile(existing, " ".join(m.get("content", "") for m in conversation if m.get("role") == "user"))
        coverage = _coverage_for(updated)
        result_profile = updated
        message = _one_follow_up(updated, coverage)
        flags: list[str] = []
        reason = "The intake profile still needs information in its weakest coverage area."
    else:
        result_profile = result.profile
        for field in PROFILE_FIELDS:
            setattr(result_profile, field, list(dict.fromkeys(getattr(existing, field) + getattr(result_profile, field))))
        result_profile.rejected_titles = list(existing.rejected_titles)  # the model must not erase these
        result_profile.history_start_index = existing.history_start_index
        result_profile.questions_asked = question_count
        coverage = result.coverage
        flags = result.flags
        message = result.message.strip()
        reason = result.reason

    # Never let a premature model status skip meaningful profile coverage.
    has_concrete_detail = sum(len(getattr(result_profile, field)) for field in PROFILE_FIELDS) >= 3
    criteria_ready = (
        coverage.interests == "solid"
        and coverage.work_style == "solid"
        and (coverage.skills_background == "solid" or coverage.values_constraints == "solid")
        and has_concrete_detail
    )
    forced_ready = _asks_for_results(state.get("message", "")) or question_count >= 8
    is_ready = forced_ready or criteria_ready
    if not is_ready:
        message = message if message and message.count("?") == 1 else _one_follow_up(result_profile, coverage)
        result_profile.questions_asked = question_count + 1
    elif not message or message.count("?") > 0:
        message = "I have enough to start exploring career directions based on what you shared."
    result_profile.coverage = coverage

    status = "ready" if is_ready else "ask"
    route = "match" if is_ready else "intake"
    print(f"[FLOW] intake -> {route} status={status} questions={result_profile.questions_asked} coverage={coverage.model_dump()}", flush=True)
    out: Flow = {"profile": result_profile.model_dump(), "coverage": coverage.model_dump(), "intake_message": message, "intake_flags": flags, "route": route, "status": status, "stage": "matching" if is_ready else "intake"}
    if is_ready:
        out["review_attempts"] = 0  # a fresh match cycle gets a fresh critic-revision budget
    return out


_REJECT_TERMS = ("don't like", "dont like", "do not like", "not interested in", "rule out", "no to ", "not for me", "not the ", "hate", "remove", "skip", "drop ")
_EXPLAIN_TERMS = ("tell me more", "more about", "what does", "what is", "what do", "how does", "how much", "what would", "?")
_RESTART_TERMS = ("start over", "start again", "begin again", "from scratch", "reset")
_ORDINALS = {"first": 0, "1st": 0, "#1": 0, "second": 1, "2nd": 1, "#2": 1, "third": 2, "3rd": 2, "#3": 2}


def _fallback_followup(message: str, known_titles: list[str], profile: Profile, entries: dict[str, dict]) -> FollowupResult:
    """Offline intent classifier for the follow-up stage."""
    text = message.lower()
    mentioned = [t for t in known_titles if t.lower() in text]
    for word, idx in _ORDINALS.items():
        if idx < len(known_titles) and known_titles[idx] not in mentioned and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text):
            mentioned.append(known_titles[idx])

    if any(term in text for term in _RESTART_TERMS):
        return FollowupResult(intent="restart")

    before = {f: list(getattr(profile, f)) for f in PROFILE_FIELDS}
    probe = _fallback_extract_profile(profile.model_copy(deep=True), message)
    updates = {f: [i for i in getattr(probe, f) if i not in before[f]] for f in PROFILE_FIELDS}
    updates = {f: v for f, v in updates.items() if v}

    if mentioned and any(term in text for term in _REJECT_TERMS):
        return FollowupResult(intent="reject", rejected_titles=mentioned, profile_updates=updates)
    if any(term in text for term in _EXPLAIN_TERMS):
        targets = mentioned or list(entries)[:1]
        parts = [f"**{t}**: {entries[t]['description']} Listed skills: {entries[t]['skills']}" for t in targets if t in entries]
        return FollowupResult(intent="explain", answer="\n\n".join(parts) or "That isn't covered in the career profiles I have.")
    if updates:
        return FollowupResult(intent="refine", profile_updates=updates)
    return FollowupResult(intent="other", answer="Would you like more detail on one of these, to rule one out, or to tell me about another priority so I can refine the suggestions?")


def followup(state: Flow) -> Flow:
    """Post-recommendation stage: reject / refine / explain / restart, looping back when needed."""
    profile = _profile_from_state(state.get("profile"))
    message = state.get("message", "")
    history = state.get("history", [])
    known_titles = [m["title"] for m in state.get("matches", []) if m.get("title")]
    docs = search_careers.invoke({"query": f"{message} {' '.join(known_titles)}".strip(), "k": 4})
    entries = {d["title"]: d for d in docs}
    valid_titles = set(known_titles) | set(entries)

    context = json.dumps({
        "conversation": history + [{"role": "user", "content": message}],
        "profile_json": profile.model_dump(),
        "known_titles": known_titles,
        "retrieved_entries": docs,
    }, ensure_ascii=False)
    result = _llm(FOLLOWUP_PROMPT, context, FollowupResult)
    if result is None:
        result = _fallback_followup(message, known_titles, profile, entries)

    # Validate model output before acting on it.
    rejected = [t for t in dict.fromkeys(result.rejected_titles) if t in valid_titles]
    updates = {f: [str(i) for i in items] for f, items in result.profile_updates.items() if f in PROFILE_FIELDS}
    intent = result.intent
    if intent == "reject" and not rejected:
        intent, result.answer = "other", "Which of the suggested careers would you like me to set aside?"
    if intent == "refine" and not any(updates.values()):
        intent, result.answer = "other", "What would you like to change or add so I can refine the suggestions?"

    out: Flow = {"followup_intent": intent, "review_attempts": 0}
    if intent == "restart":
        fresh = Profile()
        fresh.history_start_index = state.get("history_offset", 0) + len(state.get("history", []))
        # Start discovery from this message without re-extracting the old transcript.
        out.update(profile=fresh.model_dump(), matches=[], history=[], stage="intake", route="intake", status="ask")
    elif intent in ("reject", "refine"):
        for field, items in updates.items():
            setattr(profile, field, list(dict.fromkeys(getattr(profile, field) + items)))
        profile.rejected_titles = list(dict.fromkeys(profile.rejected_titles + rejected))
        out.update(profile=profile.model_dump(), route="match", status="ready", stage="matching")
    else:  # explain / other: answer from retrieved entries, stay in the recommendations stage
        answer = result.answer.strip() or "That isn't covered in the career profiles I have."
        out.update(followup_answer=answer, route="explain", stage="recommendations")
    print(f"[FLOW] followup -> {out['route']} intent={intent} rejected={rejected}", flush=True)
    return out


def matcher(state: Flow) -> Flow:
    profile = _profile_from_state(state.get("profile"))
    query = " ".join(sum((profile.interests, profile.work_style, profile.skills_background, profile.values_constraints), []))
    excluded = set(profile.rejected_titles)
    raw_docs = search_careers.invoke({"query": query, "k": 4 + len(excluded)})
    docs = [d for d in raw_docs if d["title"] not in excluded][:4]
    prompt = json.dumps({
        "profile_json": profile.model_dump(),
        "retrieved_entries_with_scores": docs,
        "excluded_titles": sorted(excluded),
        "min_score": MIN_MATCH_SCORE,
        "critic_feedback_to_address": (state.get("critique") or {}).get("consistency", {}).get("issues", []),
    }, ensure_ascii=False)
    print(f"[FLOW] matcher retrieval -> {[(d['title'], d['score']) for d in docs]} excluded={sorted(excluded)}", flush=True)
    allowed = {doc["title"]: doc for doc in docs}
    user_texts = _user_texts(state)
    if not docs or docs[0]["score"] < MIN_MATCH_SCORE:
        result = MatcherResult(no_strong_match=True, gaps_or_questions="The retrieved career profiles do not reach the minimum similarity threshold for this profile.", summary_for_user="I don’t have a strong evidence-backed match yet. A little more detail about your preferred work and priorities would help.")
    else:
        generated = _llm(MATCHER_PROMPT, prompt, MatcherResult)
        if generated is None:
            recommendations = []
            for doc in docs[:3]:
                recommendations.append(Recommendation(
                    title=doc["title"], similarity_score=doc["score"],
                    why_it_fits=f"Your profile includes {', '.join(profile.interests + profile.skills_background + profile.work_style)}. The retrieved entry describes {doc['description']} Its listed skills are {doc['skills']}.",
                    matched_user_statements=profile.interests + profile.skills_background + profile.work_style,
                    matched_entry_fields=["What they do", "Skills", "Who thrives here"],
                    possible_mismatch="The profile does not yet establish whether every listed work-style preference fits.",
                    next_step=f"Explore a small task related to {doc['title']} and compare it with the entry's listed skills.",
                ))
            result = MatcherResult(recommendations=recommendations, summary_for_user="These directions are based on the retrieved career profiles and the details you shared.")
        else:
            result = MatcherResult.model_validate(generated.model_dump())

        # Cross-validate model output against the actual retrieval evidence and the conversation.
        valid = []
        for recommendation in result.recommendations:
            doc = allowed.get(recommendation.title)
            if not doc:  # unknown, unretrieved, or rejected title
                continue
            if recommendation.matched_entry_fields and not set(recommendation.matched_entry_fields).issubset(ENTRY_FIELDS):
                continue
            valid.append(recommendation.model_copy(update={
                "similarity_score": float(doc["score"]),
                "matched_user_statements": _ground_statements(recommendation.matched_user_statements, user_texts, profile),
            }))
        if not result.no_strong_match and len(valid) < 2:
            valid = []
            for doc in docs[:2]:
                valid.append(Recommendation(
                    title=doc["title"], similarity_score=doc["score"],
                    why_it_fits=f"The profile you shared overlaps with the retrieved entry: {doc['description']} Listed skills include {doc['skills']}.",
                    matched_user_statements=profile.interests + profile.skills_background + profile.work_style,
                    matched_entry_fields=["What they do", "Skills"],
                    possible_mismatch="The match still needs checking against your priorities and constraints.",
                    next_step=f"Review the entry for {doc['title']} and tell me which listed activity sounds most or least appealing.",
                ))
        result = result.model_copy(update={"recommendations": valid[:3]})
    if result.no_strong_match:
        valid_matches = []
    else:
        valid_matches = [
            {**r.model_dump(), "rationale": r.why_it_fits, "evidence": r.matched_entry_fields}
            for r in result.recommendations
        ]
    return {"retrieved": docs, "matcher_result": result.model_dump(), "matches": valid_matches, "route": "critic", "stage": "review"}


def _grounding_issues(state: Flow) -> list[ConsistencyIssue]:
    """Deterministic checks that hold even when the critic model is offline or wrong."""
    issues: list[ConsistencyIssue] = []
    allowed = {entry["title"] for entry in state.get("retrieved", [])}
    rejected = set((state.get("profile") or {}).get("rejected_titles", []))
    for rec in (state.get("matcher_result") or {}).get("recommendations", []):
        title = rec.get("title")
        if title not in allowed:
            issues.append(ConsistencyIssue(severity="major", type="ungrounded", detail="A recommendation title is absent from retrieval.", evidence=f"{title} not in {sorted(allowed)}"))
        if title in rejected:
            issues.append(ConsistencyIssue(severity="major", type="missed_constraint", detail="The user already rejected this career.", evidence=f"{title} in rejected_titles={sorted(rejected)}"))
    return issues


def critic(state: Flow) -> Flow:
    conversation = state.get("history", []) + [{"role": "user", "content": state.get("message", "")}]
    context = json.dumps({
        "conversation": conversation,
        "profile_json": state.get("profile", {}),
        "matcher_output_json": state.get("matcher_result", {}),
        "retrieved_entries": state.get("retrieved", []),
    }, ensure_ascii=False)
    earlier_safety = SafetyReview.model_validate(state.get("safety_review") or {})
    reviewed = _llm(CRITIC_PROMPT, context, CriticResult)
    if reviewed is None:
        issues = _grounding_issues(state)
        reviewed = CriticResult(
            consistency=ConsistencyReview(verdict="revise" if issues else "pass", issues=issues),
            safety=earlier_safety,
        )
    else:
        reviewed = CriticResult.model_validate(reviewed.model_dump())
        extra = _grounding_issues(state)
        if extra:
            reviewed.consistency.issues.extend(extra)
            reviewed.consistency.verdict = "revise"
    # Never downgrade a more severe earlier safety finding.
    reviewed.safety = _merge_safety(earlier_safety, reviewed.safety)

    attempts = state.get("review_attempts", 0)
    if reviewed.safety.level == "redirect":
        route, stage = "support", "support"
    elif reviewed.consistency.verdict == "revise" and attempts < MAX_CRITIC_REVISIONS:
        route, stage = "revise", "review"
        attempts += 1
    elif reviewed.consistency.verdict == "revise":
        route, stage = "hold", "intake"
    else:
        route, stage = "final", "recommendations"
    print(f"[FLOW] critic -> {route} consistency={reviewed.consistency.verdict} safety={reviewed.safety.level} issues={len(reviewed.consistency.issues)}", flush=True)
    matches = [] if route == "hold" else state.get("matches", [])
    return {"critique": reviewed.model_dump(), "safety_review": reviewed.safety.model_dump(), "route": route, "stage": stage, "review_attempts": attempts, "matches": matches}


def respond(state: Flow) -> Flow:
    route = state.get("route")
    print(f"[FLOW] respond route={route} stage={state.get('stage', 'unchanged')}", flush=True)
    safety = SafetyReview.model_validate(state.get("safety_review", {}))
    if route == "support":
        reply = safety.message_to_user or "I’m sorry you’re dealing with this. Please reach out to someone you trust or an appropriate local professional. If you may be in immediate danger, contact local emergency services or a crisis line. We can return to career planning later."
    elif route == "pause" or (safety.level == "note" and route not in ("final",)):
        reply = safety.message_to_user or "Thanks for telling me. We can pause career questions for a moment. What kind of support would feel helpful right now?"
    elif route == "explain":
        reply = state.get("followup_answer") or "What would you like to do next with these suggestions?"
    elif route == "intake":
        reply = state.get("intake_message") or "What kind of activity do you most enjoy, and what part of it keeps your interest?"
    elif route == "hold":
        reply = "I couldn’t verify those career suggestions against everything you shared, so I’m holding them back. Which of your priorities or limits should I make sure to account for?"
    else:
        result = MatcherResult.model_validate(state.get("matcher_result", {}))
        if result.no_strong_match:
            reply = result.summary_for_user or "I don’t have a strong evidence-backed match yet. What work setting or priority would you like me to consider next?"
            if result.gaps_or_questions:
                reply += " " + result.gaps_or_questions
        else:
            rows = []
            for rec in result.recommendations:
                rows.append(f"**{rec.title}** (retrieval score {rec.similarity_score:.3f}) — {rec.why_it_fits} Possible mismatch: {rec.possible_mismatch} Next step: {rec.next_step}")
            reply = (result.summary_for_user or "Here are a few directions grounded in the career profiles I retrieved.") + "\n\n" + "\n\n".join(rows)
            if result.gaps_or_questions:
                reply += "\n\n" + result.gaps_or_questions
        if safety.level == "note":
            reply = "I hear that this has been difficult. " + reply
    return {"reply": reply, "summary": reply[:500]}


# ---------------------------------------------------------------------------
# Routing + graph
# ---------------------------------------------------------------------------
def route_after_screen(state: Flow) -> str:
    return state["route"]


def route_after_intake(state: Flow) -> str:
    return state["route"]


def route_after_followup(state: Flow) -> str:
    return state["route"]


def route_after_critic(state: Flow) -> str:
    return state["route"]


builder = StateGraph(Flow)
builder.add_node("safety_screen", safety_screen)
builder.add_node("intake", intake)
builder.add_node("followup", followup)
builder.add_node("matcher", matcher)
builder.add_node("critic", critic)
builder.add_node("respond", respond)
builder.set_entry_point("safety_screen")
builder.add_conditional_edges("safety_screen", route_after_screen, {"support": "respond", "pause": "respond", "intake": "intake", "followup": "followup"})
builder.add_conditional_edges("intake", route_after_intake, {"intake": "respond", "match": "matcher"})
builder.add_conditional_edges("followup", route_after_followup, {"match": "matcher", "explain": "respond", "intake": "intake"})
builder.add_edge("matcher", "critic")
builder.add_conditional_edges("critic", route_after_critic, {"support": "respond", "revise": "matcher", "final": "respond", "hold": "respond"})
builder.add_edge("respond", END)


def build_graph(checkpointer=None):
    """Compile the graph. Pass a LangGraph checkpointer (e.g. SqliteSaver) for native
    per-session persistence, then invoke with config={"configurable": {"thread_id": session_id}}."""
    return builder.compile(checkpointer=checkpointer)


career_graph = build_graph()
