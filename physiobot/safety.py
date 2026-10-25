"""Red-flag triage that runs before the language model.

Some symptoms need a doctor or an emergency service, not exercise advice.
Matching them with plain rules keeps the response deterministic and testable,
and means an LLM mistake can never talk someone out of seeking urgent care.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedFlag:
    name: str
    patterns: tuple[str, ...]
    advice: str


EMERGENCY = (
    "Please call 108 (ambulance) or 112, or go to the nearest hospital emergency department now. "
    "ઇમરજન્સી માટે તરત 108 પર કૉલ કરો. आपातकाल में तुरंत 108 पर कॉल करें।"
)

RED_FLAGS: tuple[RedFlag, ...] = (
    RedFlag(
        "possible heart problem",
        (
            r"chest (pain|tightness|pressure)",
            r"pain .*(left arm|jaw).*(sweat|breath)",
            r"छाती में दर्द",
            r"છાતીમાં (દુખાવો|દુઃખાવો)",
        ),
        EMERGENCY,
    ),
    RedFlag(
        "possible stroke",
        (
            r"face (droop|drooping)",
            r"slurred speech",
            r"sudden(ly)? .*(weakness|numbness|paraly)",
            r"(can'?t|cannot|unable to) (move|feel) (my )?(one side|left side|right side|arm and leg)",
            r"लकवा",
            r"લકવો",
        ),
        EMERGENCY,
    ),
    RedFlag(
        "possible spinal cord compression (cauda equina)",
        (
            r"(lost|loss of|can'?t control|no control over|leaking) .*(bladder|bowel|urine|stool|pee|poo)",
            r"(bladder|bowel|urine) .*(control|incontinen)",
            r"numb(ness)? .*(groin|saddle|genital|between (my|the) legs|private)",
            r"saddle (numbness|anaesthesia|anesthesia)",
        ),
        "Back pain together with changes in bladder or bowel control, or numbness around the groin, "
        "can mean a nerve emergency. Please go to a hospital emergency department today, or call 108.",
    ),
    RedFlag(
        "possible fracture or serious injury",
        (
            r"(bone|it) .*(sticking out|poking out)",
            r"(can'?t|cannot|unable to) (walk|stand|put (any )?weight)",
            r"(heard|felt) a (crack|snap|pop).*(swell|can'?t)",
            r"(accident|fell|fall).*(can'?t move|deform|out of place)",
        ),
        "This could be a fracture or serious injury. Please keep the area still and see a doctor or go to "
        "an emergency department for an X-ray before starting any physiotherapy. Call 108 if it is severe.",
    ),
    RedFlag(
        "possible blood clot",
        (
            r"calf .*(swollen|swelling|red|hot|warm).*(pain|tender)",
            r"(swollen|swelling) .*calf",
            r"(short(ness)? of breath|breathless).*(after|since) (surgery|operation)",
        ),
        "A painful, swollen calf, especially after surgery or long travel, can be a blood clot (DVT). "
        "Please see a doctor today. If you are also short of breath or have chest pain, call 108 now.",
    ),
    RedFlag(
        "possible infection",
        (
            r"fever .*(back|neck|joint|knee) (pain|ache)",
            r"(back|neck|joint|knee) (pain|ache) .*fever",
            r"(joint|knee) .*(hot|red) .*(swollen|swelling).*fever",
            r"stiff neck .*fever|fever .*stiff neck",
        ),
        "Pain with fever can be a sign of infection. Please see a doctor today rather than starting exercises.",
    ),
    RedFlag(
        "self-harm",
        (
            r"\b(kill myself|suicid|end my life|want to die|self[- ]harm)\b",
        ),
        "I'm really sorry you're feeling this way. You deserve support right now. Please call Tele-MANAS on "
        "14416 (free, 24x7, in Gujarati and Hindi) or 112 if you are in immediate danger.",
    ),
)

_COMPILED = tuple((flag, tuple(re.compile(p, re.IGNORECASE) for p in flag.patterns)) for flag in RED_FLAGS)


def check_red_flags(message: str) -> RedFlag | None:
    text = " ".join(message.split())
    for flag, patterns in _COMPILED:
        if any(p.search(text) for p in patterns):
            return flag
    return None


def urgent_reply(flag: RedFlag) -> str:
    return (
        f"⚠️ What you describe may be a sign of a **{flag.name}**, which needs medical attention before physiotherapy.\n\n"
        f"{flag.advice}\n\n"
        "Once a doctor has checked you, the clinic will be happy to help with your recovery."
    )
