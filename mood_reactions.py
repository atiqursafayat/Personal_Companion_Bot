"""
mood_reactions.py

Decides WHETHER and HOW the robot should react to a detected mood
(e.g. cheer up a sad user, calm down an angry one) — without reacting to
every noisy single-frame reading and without nagging the user repeatedly.

Two safeguards, same pattern discussed for battery/obstacle handling:
  - DEBOUNCE: require several consecutive matching readings before treating
    a mood as "real" (a rolling-window majority vote).
  - COOLDOWN: once a reaction fires for a given mood, don't fire again for
    that mood for a while.

HOW TO ADD A NEW MOOD REACTION
-------------------------------
Add an entry to REACTION_PROMPTS (and optionally a custom cooldown in
COOLDOWN_SECONDS). No other code changes needed — the engine picks it up
automatically.
"""

import time
from collections import deque

# How many recent mood readings to keep for the majority vote.
DEBOUNCE_WINDOW = 5
# How many of those readings must agree before a mood is considered "stable".
MIN_AGREEMENT = 3

# Per-mood cooldown after a reaction fires, in seconds. Falls back to
# DEFAULT_COOLDOWN_SECONDS for any mood not listed here.
DEFAULT_COOLDOWN_SECONDS = 300
COOLDOWN_SECONDS = {
    "SAD": 300,     # 5 min — don't keep telling someone they seem sad
    "ANGRY": 180,   # 3 min — check in a bit sooner if frustration lingers
}

# Moods that should trigger a spoken reaction, and the system-prompt hint
# used to steer the LLM's reply tone (fed into voice_assistant.get_reply).
REACTION_PROMPTS = {
    "SAD": (
        "The user appears sad or down right now. In your next reply, be "
        "extra warm and gentle. Offer a short, genuine, uplifting remark. "
        "Do not mention that you detected this from their face."
    ),
    "ANGRY": (
        "The user appears frustrated or angry right now. In your next "
        "reply, stay calm and speak gently. Briefly acknowledge their "
        "mood without being falsely cheerful or dismissive. Do not "
        "mention that you detected this from their face."
    ),
}


class MoodReactionEngine:
    def __init__(self):
        self._window = deque(maxlen=DEBOUNCE_WINDOW)
        self._last_triggered = {}  # mood -> unix timestamp

    def update_mood(self, raw_mood):
        """Feed in the latest raw mood reading (e.g. from mood_state_io.read_mood())."""
        if not raw_mood:
            return
        self._window.append(raw_mood.upper())

    @property
    def stable_mood(self):
        """The majority mood over the recent window, or None if not enough agreement yet."""
        if len(self._window) < self._window.maxlen:
            return None
        counts = {}
        for mood in self._window:
            counts[mood] = counts.get(mood, 0) + 1
        mood, count = max(counts.items(), key=lambda kv: kv[1])
        return mood if count >= MIN_AGREEMENT else None

    def check_trigger(self, is_speaking=False):
        """
        Call periodically (e.g. once per conversation loop iteration).

        Returns a dict {"mood": ..., "system_prompt": ...} if the robot
        should react right now, otherwise None.
        """
        mood = self.stable_mood
        if not mood or mood not in REACTION_PROMPTS:
            return None
        if is_speaking:
            # Don't interrupt an in-progress interaction; try again next tick.
            return None

        last_fired = self._last_triggered.get(mood, 0)
        cooldown = COOLDOWN_SECONDS.get(mood, DEFAULT_COOLDOWN_SECONDS)
        if time.time() - last_fired < cooldown:
            return None

        self._last_triggered[mood] = time.time()
        return {"mood": mood, "system_prompt": REACTION_PROMPTS[mood]}
