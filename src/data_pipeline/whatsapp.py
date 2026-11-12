"""Import, cleaning and splitting of a WhatsApp group conversation export.

Supported inputs:
- WhatsApp plain-text export (Android/iOS-style timestamps), even if saved as conv_whatsapp.json;
- JSON list or object containing `messages` / `chat` / `data`;
- JSON records with sender + text/content/body/message and optional id/reply_to/parent_id;
- the repository's existing `{"conversations": [{role, content}, ...]}` format;
- explicit `{parent: ..., child: ...}` pairs.

No third-party dependency is required. Raw participant names are never written to the
resulting CPT/SFT datasets. This is pseudonymisation, not a guarantee of anonymity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# WhatsApp Android and iOS export formats. The date/time is parsed separately so
# we can make a conservative adjacent-message SFT candidate only when a timestamp exists.
_ANDROID_RE = re.compile(
    r"^(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)\s+-\s+(?P<rest>.*)$",
    re.IGNORECASE,
)
_IOS_RE = re.compile(
    r"^\[(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{2,4})[, ]+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)\]\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

_MEDIA_PLACEHOLDERS = re.compile(
    r"^(?:<media omitted>|<médias omis>|<media omis>|"
    r"<(?:image|video|vidéo|sticker|audio|document|gif) omitted>|"
    r"image omitted|video omitted|vidéo omise|sticker omitted|"
    r"audio omitted|document omitted|gif omitted|"
    r"image non incluse|vidéo non incluse|sticker non inclus|"
    r"fichier joint|pièce jointe|attached file)(?:\s*\(.*\))?\s*$",
    re.IGNORECASE,
)
_REACTION_EVENT = re.compile(
    r"(?:reacted\s+.{0,30}\s+to\s+|a\s+réagi\s+.{0,30}\s+à\s+|"
    r"a\s+réagi\s+avec\s+.{0,30}\s+à\s+)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s()./-]{6,}\d(?!\w)")
_MENTION_RE = re.compile(r"(?<!\w)@[A-Za-zÀ-ÖØ-öø-ÿ0-9_.-]+")
_HASH_TAG_RE = re.compile(r"(?<!\w)#[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")

# Broad Unicode emoji/symbol ranges. Punctuation and ordinary accented letters are preserved.
_EMOJI_RE = re.compile(
    "["
    "\\U0001F1E6-\\U0001F1FF"
    "\\U0001F300-\\U0001FAFF"
    "\\U00002600-\\U000026FF"
    "\\U00002700-\\U000027BF"
    "\\U0000FE0F\\U0000200D\\U000020E3"
    "]+",
    flags=re.UNICODE,
)

_SYSTEM_TEXT = (
    "messages and calls are end-to-end encrypted",
    "les messages et les appels sont chiffrés de bout en bout",
    "ce message a été supprimé",
    "vous avez supprimé ce message",
    "this message was deleted",
    "you deleted this message",
)


@dataclass
class Message:
    message_id: str
    sender: str
    text: str
    timestamp: Optional[datetime]
    reply_to: Optional[str]
    order: int


def _parse_datetime(date_text: str, time_text: str) -> Optional[datetime]:
    raw = f"{date_text.strip()} {time_text.strip()}"
    formats = (
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%y %H:%M:%S", "%m/%d/%y %H:%M",
        "%d/%m/%Y %I:%M %p", "%d/%m/%y %I:%M %p", "%m/%d/%Y %I:%M %p", "%m/%d/%y %I:%M %p",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%y %H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    )
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _parse_whatsapp_text(text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for line in text.splitlines():
        match = _ANDROID_RE.match(line) or _IOS_RE.match(line)
        if match:
            rest = match.group("rest")
            timestamp = _parse_datetime(match.group("date"), match.group("time"))
            # A sender is separated from the message by the first ': '. System notifications
            # often have no sender separator and are intentionally ignored.
            if ": " not in rest:
                current = None
                continue
            sender, body = rest.split(": ", 1)
            sender = sender.strip()
            if not sender:
                current = None
                continue
            current = {
                "id": f"txt-{len(records)}",
                "sender": sender,
                "text": body,
                "timestamp": timestamp,
            }
            records.append(current)
        elif current is not None:
            # WhatsApp messages can span multiple lines.
            current["text"] = f"{current['text']}\n{line}".strip()

    return records


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _normalise_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str) and value.strip():
        candidate = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(candidate).replace(tzinfo=None)
        except ValueError:
            pass
        for fmt in (
            "%d/%m/%Y, %H:%M:%S", "%d/%m/%Y, %H:%M", "%d/%m/%y, %H:%M",
            "%m/%d/%Y, %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        ):
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def _get_reply_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("reply_to", "replyTo", "reply_to_id", "parent_id", "parentId", "quotedMessageId"):
        value = record.get(key)
        if isinstance(value, dict):
            value = value.get("id") or value.get("message_id")
        if value is not None and str(value).strip():
            return str(value).strip()
    context = record.get("contextInfo")
    if isinstance(context, dict):
        value = context.get("stanzaId") or context.get("quotedMessageId")
        if value:
            return str(value)
    return None


def _record_to_message(record: Dict[str, Any], order: int) -> Optional[Dict[str, Any]]:
    msg_type = str(record.get("type") or record.get("message_type") or record.get("media_type") or "").lower()
    if any(word in msg_type for word in ("reaction", "sticker", "video", "image", "audio", "document", "gif")):
        return None

    body = ""
    for key in ("text", "content", "body", "message", "caption"):
        if key in record:
            body = _as_text(record.get(key))
            if body:
                break
    if not body:
        return None

    sender_value = record.get("sender") or record.get("author") or record.get("from") or record.get("participant") or record.get("name")
    if isinstance(sender_value, dict):
        sender_value = sender_value.get("name") or sender_value.get("id") or sender_value.get("phone")
    sender = _as_text(sender_value).strip() or f"unknown-{order}"
    msg_id = record.get("id") or record.get("message_id") or record.get("key") or f"json-{order}"
    if isinstance(msg_id, dict):
        msg_id = msg_id.get("id") or msg_id.get("_serialized") or f"json-{order}"

    return {
        "id": str(msg_id),
        "sender": sender,
        "text": body,
        "timestamp": _normalise_timestamp(record.get("timestamp") or record.get("date") or record.get("datetime") or record.get("time")),
        "reply_to": _get_reply_id(record),
    }


def _extract_json_records(data: Any) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Return flat message records plus already-structured user/assistant pairs."""
    raw_records: List[Dict[str, Any]] = []
    direct_pairs: List[Tuple[str, str]] = []

    def add_message_tree(item: Dict[str, Any], order: int, inherited_parent_id: Optional[str] = None) -> None:
        """Flatten nested replies/children while preserving an explicit parent link."""
        record_input = dict(item)
        if inherited_parent_id and not _get_reply_id(record_input):
            record_input["reply_to"] = inherited_parent_id
        record = _record_to_message(record_input, order)
        current_id = None
        if record:
            raw_records.append(record)
            current_id = str(record["id"])
        else:
            candidate_id = record_input.get("id") or record_input.get("message_id")
            current_id = str(candidate_id) if candidate_id is not None else inherited_parent_id

        if current_id:
            for child_key in ("replies", "children", "answers"):
                children = record_input.get(child_key)
                if isinstance(children, list):
                    for child_index, child in enumerate(children):
                        if isinstance(child, dict):
                            add_message_tree(child, order * 1000 + child_index + 1, current_id)

    if isinstance(data, dict):
        # Existing repo-style SFT records.
        conversations = data.get("conversations")
        if isinstance(conversations, list) and conversations:
            user_text = None
            assistant_text = None
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role", "")).lower()
                content = _as_text(turn.get("content"))
                if role in ("user", "human", "parent") and user_text is None:
                    user_text = content
                elif role in ("assistant", "bot", "child") and assistant_text is None:
                    assistant_text = content
            if user_text and assistant_text:
                direct_pairs.append((user_text, assistant_text))
                raw_records.extend([
                    {"id": f"conv-user-{len(direct_pairs)}", "sender": "user", "text": user_text},
                    {"id": f"conv-assistant-{len(direct_pairs)}", "sender": "assistant", "text": assistant_text},
                ])
            return raw_records, direct_pairs

        # Explicit parent/child JSON pair.
        if "parent" in data and "child" in data:
            parent, child = data.get("parent"), data.get("child")
            p_text = _as_text(parent.get("text") or parent.get("content") or parent.get("message")) if isinstance(parent, dict) else _as_text(parent)
            c_text = _as_text(child.get("text") or child.get("content") or child.get("message")) if isinstance(child, dict) else _as_text(child)
            if p_text and c_text:
                direct_pairs.append((p_text, c_text))
                raw_records.extend([
                    {"id": f"pair-parent-{len(direct_pairs)}", "sender": "parent", "text": p_text},
                    {"id": f"pair-child-{len(direct_pairs)}", "sender": "child", "text": c_text},
                ])
            return raw_records, direct_pairs

        for key in ("messages", "chat", "data", "items", "results"):
            if isinstance(data.get(key), list):
                return _extract_json_records(data[key])
        # Single message object, including any nested replies/children.
        add_message_tree(data, 0)
        return raw_records, direct_pairs

    if isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict):
                # Nested explicit pairs.
                if "parent" in item and "child" in item:
                    p, c = item.get("parent"), item.get("child")
                    p_text = _as_text(p.get("text") or p.get("content") or p.get("message")) if isinstance(p, dict) else _as_text(p)
                    c_text = _as_text(c.get("text") or c.get("content") or c.get("message")) if isinstance(c, dict) else _as_text(c)
                    if p_text and c_text:
                        direct_pairs.append((p_text, c_text))
                        raw_records.extend([
                            {"id": f"pair-parent-{index}", "sender": "parent", "text": p_text},
                            {"id": f"pair-child-{index}", "sender": "child", "text": c_text},
                        ])
                    continue
                # A list of repo-style conversation records.
                if isinstance(item.get("conversations"), list):
                    pair_records, pairs = _extract_json_records(item)
                    raw_records.extend(pair_records)
                    direct_pairs.extend(pairs)
                    continue
                add_message_tree(item, index)
            elif isinstance(item, str):
                # A bare-string list is accepted as isolated CPT text, not guessed into SFT.
                raw_records.append({"id": f"string-{index}", "sender": f"unknown-{index}", "text": item})
        return raw_records, direct_pairs

    raise ValueError("Format JSON WhatsApp non reconnu : un objet ou une liste était attendu.")


def _looks_like_non_text_payload(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _MEDIA_PLACEHOLDERS.match(stripped):
        return True
    if _REACTION_EVENT.search(stripped):
        return True
    low = stripped.lower()
    if any(low.startswith(prefix) for prefix in _SYSTEM_TEXT):
        return True
    # WhatsApp exports can spell attachment placeholders in English or French.
    if re.fullmatch(r"\[(?:sticker|video|image|gif|audio|document)(?: omitted| non inclus| omis)?\]", low):
        return True
    return False


_PARTICIPANT_PATTERN_CACHE: Dict[Tuple[str, ...], List[Tuple[re.Pattern, str]]] = {}


def _get_participant_patterns(participant_names: Sequence[str]) -> List[Tuple[re.Pattern, str]]:
    key = tuple(participant_names)
    cached = _PARTICIPANT_PATTERN_CACHE.get(key)
    if cached is not None:
        return cached
    names = sorted({name.strip() for name in participant_names if name and len(name.strip()) >= 2}, key=len, reverse=True)
    patterns = [
        (re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE), f"[PERSONNE_{index:02d}]")
        for index, name in enumerate(names, start=1)
    ]
    _PARTICIPANT_PATTERN_CACHE[key] = patterns
    return patterns


def clean_message(text: str, participant_names: Sequence[str] = ()) -> str:
    """Remove links, emoji, tags and common personal identifiers."""
    cleaned = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", str(text or ""))
    if _looks_like_non_text_payload(cleaned):
        return ""

    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _EMAIL_RE.sub(" [EMAIL] ", cleaned)
    cleaned = _PHONE_RE.sub(" [NUMERO] ", cleaned)

    # Replace known group-member names wherever they occur, including tagged mentions.
    for pattern, replacement in _get_participant_patterns(participant_names):
        cleaned = pattern.sub(replacement, cleaned)

    cleaned = re.sub(r"@\s*(?=\[(?:PERSONNE_\d+|NUMERO|EMAIL)\])", "", cleaned)
    cleaned = _MENTION_RE.sub(" [MENTION] ", cleaned)
    cleaned = _HASH_TAG_RE.sub(" ", cleaned)
    cleaned = _EMOJI_RE.sub(" ", cleaned)
    cleaned = re.sub(r"(.)\1{3,}", r"\1\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Drop remaining emoji-only / punctuation-only / link-only messages.
    if not re.search(r"[A-Za-zÀ-ÿ0-9]", cleaned):
        return ""
    return cleaned


def _build_speaker_aliases(messages: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    role_labels = {"user", "assistant", "bot", "parent", "child", "human"}
    for message in messages:
        sender = str(message.get("sender", "")).strip()
        # Canonical role labels are not real participant names and should not be masked
        # inside ordinary prose by accident.
        if sender and sender.casefold() not in role_labels and sender not in aliases:
            aliases[sender] = f"Participant_{len(aliases) + 1:03d}"
    return aliases


def _build_sft_pairs(messages: Sequence[Message], direct_pairs: Sequence[Tuple[str, str]], max_gap_minutes: int) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    by_id = {message.message_id: message for message in messages}

    # Explicit quote/reply relationships take priority.
    for child in messages:
        parent = by_id.get(child.reply_to or "")
        if parent is not None and parent.message_id != child.message_id and parent.sender != child.sender:
            pairs.append({"user": parent.text, "assistant": child.text, "source": "whatsapp_explicit_reply"})

    # Existing conversational-pair JSON format is retained as direct supervision.
    for parent_text, child_text in direct_pairs:
        pairs.append({"user": parent_text, "assistant": child_text, "source": "whatsapp_structured_pair"})

    # Plain-text exports normally lose quote/reply metadata. Use only immediate adjacent
    # turns by different people within a short time window; never infer a reply if timestamps
    # are missing. This is a heuristic, so it is clearly marked in `source`.
    ordered = sorted(messages, key=lambda m: m.order)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.sender == current.sender or previous.timestamp is None or current.timestamp is None:
            continue
        gap = (current.timestamp - previous.timestamp).total_seconds()
        if 0 <= gap <= max_gap_minutes * 60:
            pairs.append({"user": previous.text, "assistant": current.text, "source": "whatsapp_adjacent_heuristic"})

    return pairs


def load_whatsapp_dataset(
    path: Path,
    lexicon: Dict[str, Any],
    *,
    cpt_threshold: float = 0.03,
    cpt_min_words: int = 3,
    sft_threshold: float = 0.06,
    max_gap_minutes: int = 3,
) -> Tuple[List[str], List[Dict[str, str]], Dict[str, int]]:
    """Load a WhatsApp export and return (CPT texts, SFT candidates, statistics).

    CPT keeps messages scoring at least `cpt_threshold`, with a lower 3-word minimum
    than the existing YouTube path so shorter natural group-chat turns are not all lost.
    SFT pairs are returned unfiltered; the main pipeline applies the project's existing
    stricter assistant score and dialogue checks.
    """
    path = Path(path)
    stats = {
        "messages_loaded": 0,
        "messages_cleaned": 0,
        "cpt_kept": 0,
        "sft_candidates": 0,
        "direct_pairs": 0,
    }
    if not path.exists():
        return [], [], stats

    raw = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw)
        raw_records, direct_pairs = _extract_json_records(payload)
    except json.JSONDecodeError:
        raw_records = _parse_whatsapp_text(raw)
        direct_pairs = []

    stats["messages_loaded"] = len(raw_records)
    stats["direct_pairs"] = len(direct_pairs)
    aliases = _build_speaker_aliases(raw_records)
    participant_names = list(aliases.keys())

    messages: List[Message] = []
    seen_ids = set()
    for order, record in enumerate(raw_records):
        raw_text = _as_text(record.get("text"))
        text = clean_message(raw_text, participant_names)
        if not text:
            continue
        stats["messages_cleaned"] += 1
        message_id = str(record.get("id") or f"wa-{order}")
        if message_id in seen_ids:
            message_id = f"{message_id}-{order}"
        seen_ids.add(message_id)
        sender_raw = str(record.get("sender", "unknown"))
        messages.append(
            Message(
                message_id=message_id,
                sender=aliases.get(sender_raw, "Participant_Unknown"),
                text=text,
                timestamp=record.get("timestamp") if isinstance(record.get("timestamp"), datetime) else _normalise_timestamp(record.get("timestamp")),
                reply_to=str(record.get("reply_to")) if record.get("reply_to") is not None else None,
                order=order,
            )
        )

    # Sanitize direct pairs, including mentions.
    clean_direct_pairs = [
        (clean_message(parent, participant_names), clean_message(child, participant_names))
        for parent, child in direct_pairs
    ]
    clean_direct_pairs = [(p, c) for p, c in clean_direct_pairs if p and c]

    # Reuse the project's scorer for Camfranglais relevance, with a more permissive CPT
    # threshold/minimum length for spontaneous short group messages.
    from src.data_pipeline.scorer import score_camfranglais

    cpt_texts: List[str] = []
    seen_texts = set()
    for message in messages:
        score = score_camfranglais(
            message.text,
            lexicon,
            min_words=cpt_min_words,
            high_threshold=cpt_threshold,
            grey_threshold=max(0.0, cpt_threshold / 2),
        )
        cleaned = score["cleaned_text"]
        key = cleaned.casefold()
        if score["score"] >= cpt_threshold and key not in seen_texts:
            seen_texts.add(key)
            cpt_texts.append(cleaned)

    sft_pairs = _build_sft_pairs(messages, clean_direct_pairs, max_gap_minutes)
    stats["cpt_kept"] = len(cpt_texts)
    stats["sft_candidates"] = len(sft_pairs)
    return cpt_texts, sft_pairs, stats
