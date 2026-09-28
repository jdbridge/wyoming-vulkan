"""Text preparation, ported from `pocket_tts.models.text_chunking`.

The model is trained on single sentences, so a long input is split on sentence
boundaries and regrouped into chunks that fit `max_tokens`.
"""

import logging
import re

logger = logging.getLogger(__name__)


# A blank line is a sentence boundary; a single line break is not. Newlines are
# flattened to spaces before splitting, so a title or a paragraph that ends
# without punctuation ran straight into the next line. Only a blank line is
# unambiguous enough to act on: it gets a period when the line before it ends
# in a letter or digit, and a single break is left alone, since a sentence that
# merely wraps must not be cut.
_PARAGRAPH = re.compile(r"([^\s\"'\u201d\u2019)\]])([\"'\u201d\u2019)\]]*)[ \t]*\n[ \t]*\n\s*")


def break_paragraphs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        last, closers = match.group(1), match.group(2)
        return f"{last}{closers}. " if last.isalnum() else f"{last}{closers} "

    return _PARAGRAPH.sub(repl, text)


def prepare_text_prompt(
    text: str, pad_with_spaces_for_short_inputs: bool, remove_semicolons: bool
) -> tuple[str, int]:
    """Normalise a chunk and guess how many frames to keep past EOS."""
    text = break_paragraphs(text.strip())
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if remove_semicolons:
        text = text.replace(";", ",")
    frames_after_eos_guess = 3 if len(text.split()) <= 4 else 1

    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text[-1].isalnum():
        text = text + "."
    if pad_with_spaces_for_short_inputs and len(text.split()) < 5:
        text = " " * 8 + text
    return text, frames_after_eos_guess


def _is_decimal_period_boundary(tokens: list[int], start: int, sp) -> bool:
    prefix = sp.decode(tokens[:start])
    suffix = sp.decode(tokens[start:])
    return (
        len(prefix) >= 2
        and prefix[-1] == "."
        and prefix[-2].isdigit()
        and bool(suffix)
        and suffix[0].isdigit()
    )


def _boundary_indices(
    tokens: list[int], boundary_tokens: list[int], sp=None, skip_decimal_periods: bool = False
) -> list[int]:
    boundary_set = set(boundary_tokens)
    indices = [0]
    previous_was_boundary = False
    for idx, token in enumerate(tokens):
        if token in boundary_set:
            previous_was_boundary = True
            continue
        if previous_was_boundary:
            if skip_decimal_periods and sp is not None and _is_decimal_period_boundary(
                tokens, idx, sp
            ):
                previous_was_boundary = False
                continue
            indices.append(idx)
        previous_was_boundary = False
    indices.append(len(tokens))
    return indices


def _segments(tokens: list[int], boundaries: list[int], sp) -> list[tuple[int, str]]:
    return [
        (end - start, sp.decode(tokens[start:end]))
        for start, end in zip(boundaries, boundaries[1:])
    ]


def split_into_best_sentences(
    sp,
    text: str,
    max_tokens: int,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
) -> list[str]:
    text, _ = prepare_text_prompt(text, pad_with_spaces_for_short_inputs, remove_semicolons)
    tokens = sp.encode(text.strip(), out_type=int)

    _, *sentence_ends = sp.encode(".!...?", out_type=int)
    boundaries = _boundary_indices(tokens, sentence_ends, sp, skip_decimal_periods=True)
    sentences = _segments(tokens, boundaries, sp)

    # Sub-split oversized sentences on commas, semicolons and colons, otherwise
    # the model tends to skip words.
    _, *fallback = sp.encode(",;:", out_type=int)
    refined: list[tuple[int, str]] = []
    for count, sentence in sentences:
        if count <= max_tokens:
            refined.append((count, sentence))
            continue
        sub_tokens = sp.encode(sentence.strip(), out_type=int)
        sub = _segments(sub_tokens, _boundary_indices(sub_tokens, fallback), sp)
        refined.extend(sub if len(sub) > 1 else [(count, sentence)])

    chunks: list[str] = []
    current, current_count = "", 0
    for count, sentence in refined:
        if current == "":
            current, current_count = sentence, count
        elif current_count + count > max_tokens:
            chunks.append(current.strip())
            current, current_count = sentence, count
        else:
            current += " " + sentence
            current_count += count
    if current != "":
        chunks.append(current.strip())

    for chunk in chunks:
        if len(sp.encode(chunk.strip(), out_type=int)) > max_tokens:
            logger.warning(
                "Chunk has more than %d tokens, generation may skip words: '%.50s...'",
                max_tokens,
                chunk,
            )
    return chunks


_MARKER_RE = re.compile(r"<IPA_U([0-9A-Fa-f]{4,6})>")
_SENTENCE_END = ".!?"
_CLAUSE_END = ",;:"


class MixedTokenizer:
    """SentencePiece for text and punctuation, one id per IPA character.

    Ported from the adapter's training branch. Ids run `0..vocab_base-1` for the
    pretrained pieces, then one row per character of the IPA inventory, then a
    pad row. Anything that is not IPA — punctuation, spaces, digits, ordinary
    letters — keeps going through SentencePiece, which is what lets a single
    string mix Hebrew phonemes with English words.
    """

    def __init__(self, sp, ipa_chars: str, vocab_base: int):
        self.sp = sp
        self.ipa_chars = ipa_chars
        self.base = vocab_base
        self.char_to_id = {char: vocab_base + i for i, char in enumerate(ipa_chars)}
        self.ipa_set = set(ipa_chars)
        self.pad_id = vocab_base + len(ipa_chars)

    def encode(self, text: str, out_type=int) -> list[int]:
        text = _MARKER_RE.sub(lambda match: chr(int(match.group(1), 16)), text)
        ids: list[int] = []
        pending: list[str] = []

        def flush() -> None:
            if not pending:
                return
            run = "".join(pending)
            pieces = self.sp.encode(run)
            if not pieces and run.strip() == "":
                pieces = [self.sp.piece_to_id("▁")]  # a bare space
            ids.extend(pieces)
            pending.clear()

        for char in text:
            if char in self.ipa_set:
                flush()
                ids.append(self.char_to_id[char])
            else:
                pending.append(char)
        flush()
        return ids


def prepare_phoneme_prompt(text: str) -> str:
    """Whitespace tidying only.

    The English path uppercases the first character and appends a period; both
    corrupt IPA, so the adapter's inference path skips them and so do we.
    """
    text = break_paragraphs(text.strip()).replace("\n", " ").replace("\r", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    return text


def _split_keeping_delimiters(text: str, delimiters: str) -> list[str]:
    segments, current = [], ""
    for char in text:
        current += char
        if char in delimiters:
            continue
        if current[:-1] and current[-2] in delimiters:
            segments.append(current[:-1].strip())
            current = char
    if current.strip():
        segments.append(current.strip())
    return [segment for segment in segments if segment]


def split_phoneme_chunks(tokenizer: MixedTokenizer, text: str, max_tokens: int) -> list[str]:
    """Leave short input alone; split longer input on punctuation.

    A phoneme string that already fits is passed through untouched, which is how
    the adapter was trained and evaluated. Only once it exceeds the model's
    per-chunk budget do we fall back to splitting, on sentence enders first and
    then on clause punctuation.
    """
    text = prepare_phoneme_prompt(text)
    if len(tokenizer.encode(text)) <= max_tokens:
        return [text]

    segments: list[str] = []
    for sentence in _split_keeping_delimiters(text, _SENTENCE_END):
        if len(tokenizer.encode(sentence)) <= max_tokens:
            segments.append(sentence)
        else:
            segments.extend(_split_keeping_delimiters(sentence, _CLAUSE_END))

    chunks: list[str] = []
    current, current_count = "", 0
    for segment in segments:
        count = len(tokenizer.encode(segment))
        if current == "":
            current, current_count = segment, count
        elif current_count + count > max_tokens:
            chunks.append(current)
            current, current_count = segment, count
        else:
            current += " " + segment
            current_count += count
    if current:
        chunks.append(current)

    for chunk in chunks:
        if len(tokenizer.encode(chunk)) > max_tokens:
            logger.warning(
                "Phoneme chunk has more than %d tokens and has no punctuation to "
                "split on, generation may skip sounds: '%.40s...'",
                max_tokens,
                chunk,
            )
    return chunks
