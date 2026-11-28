"""The Discord and Telegram renderers strip an unclosed steer marker in linear time.

Each renderer removes a trailing ``[STEERING …`` fragment that is still streaming.
The original tail literals, kept below as the oracle and run on the whole text as
before, define the result; the renderers must return the same text for every input.
The tail literal must read only the text after the last character that can end a
match, so the openers in front of a closing bracket are never scanned again on each
streamed frame.
"""

from __future__ import annotations

import random
import re

import pytest

from kiro_crew.discord import renderer_text as discord_text
from kiro_crew.messaging.steer_ack import strip_steering_acks
from kiro_crew.telegram import renderer_display as telegram_display

#: Each renderer's unclosed-frame literal before the tail was bounded, oracle only.
OLD_DISCORD_TAIL = re.compile(r"\[STEERING\b[^\]\r\n]*$")
OLD_TELEGRAM_TAIL = re.compile(r"\[STEERING\b[^\]]*$")


def oracle_discord(text: str) -> str:
    cleaned = OLD_DISCORD_TAIL.sub("", strip_steering_acks(text))
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def oracle_telegram(text: str) -> str:
    cleaned = OLD_TELEGRAM_TAIL.sub("", strip_steering_acks(text))
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


RENDERERS = [
    pytest.param(discord_text._strip_steering, oracle_discord, id="discord"),
    pytest.param(telegram_display._strip_steering, oracle_telegram, id="telegram"),
]

PINNED = [
    "",
    "plain text",
    "hello [STEERING pause the plan",
    "hello [STEERING pause the plan\n",
    "hello [STEERING pause the plan\n\n",
    "hello [STEERING pause the plan\r\n",
    "hello [STEERING pause the plan\r",
    "hello [STEERING pause\nthe plan",
    "hello [STEERING",
    "hello [STEERING\n",
    "[STEERINGX tail",
    "[STEERING_ tail",
    "[STEERING\u00e9 tail",
    "[STEERING steer-4a2f: x] then [STEERING steer-9b1c: y",
    "[STEERING steer-4a2f: x]\n\n\n\nafter",
    "[STEERING a [STEERING b] c",
    "[STEERING a [STEERING b",
    "[STEERING a [STEERING b] [STEERING c",
    "[STEERING a [STEERING b\n[STEERING c]",
    "[STEERING a\n[STEERING b",
    "[STEERING a\n[STEERING b\n",
    "[STEERING a\r[STEERING b\r\n",
    "[STEERING a]\n",
    "[STEERING a\n]\n",
    "x\n\n\n\n[STEERING a",
    "[STEERING a" * 5 + "]",
    "[STEERING a" * 5 + "\n",
    "[STEERING a" * 5 + "]\n",
]

TOKENS = [
    "[STEERING",
    "[STEERING ",
    "[STEERING a",
    "[STEERINGX",
    "[STEERING_",
    "[STEERING\u00e9",
    "[STEERING steer-4a2f: x]",
    "[STEERING steer-4a2f: x",
    "]",
    "[",
    "\n",
    "\n\n",
    "\r",
    "\r\n",
    " ",
    "text",
    "a b",
]


def _random_texts(count: int) -> list[str]:
    rng = random.Random(20261128)
    return ["".join(rng.choice(TOKENS) for _ in range(rng.randint(0, 9))) for _ in range(count)]


@pytest.mark.parametrize(("renderer", "oracle"), RENDERERS)
@pytest.mark.parametrize("text", PINNED)
def test_a_pinned_input_renders_as_the_original_tail_did(renderer, oracle, text):
    assert renderer(text) == oracle(text)


@pytest.mark.parametrize(("renderer", "oracle"), RENDERERS)
def test_seeded_random_inputs_render_as_the_original_tail_did(renderer, oracle):
    texts = _random_texts(3000)
    assert len(texts) >= 3000
    for text in texts:
        assert renderer(text) == oracle(text), repr(text)


class _RecordingRe:
    """Stands in for a renderer module's ``re`` and records what each ``sub`` reads."""

    def __init__(self) -> None:
        self.reads: list[tuple[str, str]] = []

    def sub(self, pattern, repl, string, *args, **kwargs):
        self.reads.append((pattern, string))
        return re.sub(pattern, repl, string, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(re, name)


TAIL_READERS = [
    pytest.param(discord_text, OLD_DISCORD_TAIL.pattern, oracle_discord, id="discord"),
    pytest.param(telegram_display, OLD_TELEGRAM_TAIL.pattern, oracle_telegram, id="telegram"),
]


@pytest.mark.parametrize(("module", "tail", "oracle"), TAIL_READERS)
@pytest.mark.parametrize("openers", [1, 50, 2000])
def test_the_tail_literal_reads_none_of_the_openers_before_a_final_bracket(
    monkeypatch, module, tail, oracle, openers
):
    recorder = _RecordingRe()
    monkeypatch.setattr(module, "re", recorder)
    text = "[STEERING a" * openers + "]"

    assert module._strip_steering(text) == oracle(text) == text
    # No match can start before the last ``]``, so the literal gets nothing to scan,
    # however many openers sit in front of the bracket.
    assert [read for pattern, read in recorder.reads if pattern == tail] == [""]


@pytest.mark.parametrize(("module", "tail", "oracle"), TAIL_READERS)
def test_the_tail_literal_reads_only_the_fragment_after_the_last_bracket(
    monkeypatch, module, tail, oracle
):
    recorder = _RecordingRe()
    monkeypatch.setattr(module, "re", recorder)
    text = "[STEERING a" * 2000 + "] reply [STEERING pause the"
    rendered = module._strip_steering(text)

    assert rendered == oracle(text)
    assert "pause" not in rendered
    assert [read for pattern, read in recorder.reads if pattern == tail] == [
        " reply [STEERING pause the"
    ]
