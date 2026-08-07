import os
import random
import shutil
import subprocess
import sys
import time

from groq import Groq
import pygame
import sounddevice as sd
import soundfile as sf

from assistant_tools import get_assistant_volume
from voice_commands import answer_local_question
import robot_commands  # noqa: F401 -- import registers I'm-okay/help/temperature/status commands
from alert_logic import set_alert_state
from event_log_io import log_event
from mood_reactions import MoodReactionEngine
from mood_state_io import read_mood
from personal_memory import describe_memory, read_memory
from reminder_engine import get_due_reminders, publish_reminder_state
from robot_state_io import read_field, write_field

# Initialize pygame mixer once at startup
pygame.mixer.init()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
VOICE_MODEL = (
    "piper_voices/en_US-amy-medium.onnx"
)
WAKE_PHRASES = ("wake up",)
STOP_PHRASES = ("stop",)
SLEEP_LISTEN_SECONDS = 3
ACTIVE_LISTEN_SECONDS = 5
PREFERRED_SAMPLE_RATES = (16000, 22050, 44100, 48000)


def set_voice_status(status):
    """Best-effort voice status publish for the dashboard."""
    try:
        write_field("voice_status", status)
    except Exception:
        # Voice loop should continue even if shared-state write fails.
        pass


def set_assistant_output(text):
    """Best-effort publish of the assistant's latest spoken output."""
    try:
        write_field("assistant_output", text)
    except Exception:
        # Voice loop should continue even if shared-state write fails.
        pass


def normalize_text(text):
    return " ".join(text.lower().strip().split())


def has_phrase(text, phrases):
    normalized = normalize_text(text)
    return any(phrase in normalized for phrase in phrases)


def _resolve_input_device():
    default_device = sd.default.device
    input_device = (
        default_device[0]
        if isinstance(default_device, (list, tuple))
        else default_device
    )

    if input_device not in (None, -1):
        try:
            device_info = sd.query_devices(input_device, kind="input")
            if device_info["max_input_channels"] > 0:
                return input_device, device_info
        except Exception:
            pass

    for device_index, device_info in enumerate(sd.query_devices()):
        if device_info["max_input_channels"] > 0:
            return device_index, device_info

    raise RuntimeError("No usable audio input device was found.")


def _choose_samplerate(device_index, device_info):
    for samplerate in PREFERRED_SAMPLE_RATES:
        try:
            sd.check_input_settings(
                device=device_index, samplerate=samplerate, channels=1
            )
            return samplerate
        except Exception:
            continue

    fallback_rate = int(round(device_info.get("default_samplerate") or 48000))
    sd.check_input_settings(device=device_index, samplerate=fallback_rate, channels=1)
    return fallback_rate


def record_audio(filename="input.wav", duration=5, samplerate=None):
    device_index, device_info = _resolve_input_device()
    samplerate = samplerate or _choose_samplerate(device_index, device_info)

    print(f"Recording from {device_info['name']} at {samplerate} Hz...")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        device=device_index,
        dtype="float32",
    )
    sd.wait()

    sf.write(filename, audio[:, 0], samplerate)


def transcribe(filename="input.wav"):
    with open(filename, "rb") as f:
        transcript = client.audio.transcriptions.create(
            file=f, model="whisper-large-v3-turbo"
        )
    return transcript.text


def get_reply(user_text, history, extra_system_context=None):
    history.append({"role": "user", "content": user_text})

    system_content = (
        "You are a warm, caring companion robot. Keep replies to"
        " 1-2 short sentences."
    )
    memory_summary = describe_memory()
    if memory_summary:
        system_content += " Personal memory: " + memory_summary
    if extra_system_context:
        system_content += " " + extra_system_context

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_content},
            *history,
        ],
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


def speak(text, filename="reply.wav"):
    piper_command = shutil.which("piper")
    if not piper_command:
        piper_command = os.path.join(os.path.dirname(sys.executable), "piper")

    if not os.path.exists(piper_command):
        raise FileNotFoundError(
            "Could not find the Piper executable. Activate the venv or install piper-tts."
        )

    subprocess.run(
        [piper_command, "--model", VOICE_MODEL, "--output_file", filename],
        input=text.encode(),
        check=True,
    )

    pygame.mixer.music.load(filename)
    pygame.mixer.music.set_volume(get_assistant_volume() / 100.0)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.01)


def listen_for_wake_word():
    set_voice_status("sleep-listening")
    print("Sleep mode. Say 'Wake up' to activate.")
    while True:
        record_audio(duration=SLEEP_LISTEN_SECONDS, filename="sleep.wav")
        set_voice_status("transcribing")
        text = transcribe("sleep.wav")
        if not text:
            set_voice_status("sleep-listening")
            continue

        print("Heard:", text)
        if has_phrase(text, WAKE_PHRASES):
            set_assistant_output("I'm awake.")
            set_voice_status("speaking")
            speak("I'm awake.")
            set_voice_status("awake")
            return

        set_voice_status("sleep-listening")


mood_engine = MoodReactionEngine()

ACTION_PHRASES = {
    "offer_encouragement": [
        "I'm proud of you for getting through this moment.",
        "You deserve a little gentleness right now.",
    ],
    "suggest_happy_memory": [
        "Maybe think of one memory that usually makes you smile.",
        "If you want, we can hold onto one small good memory together.",
    ],
    "offer_gentle_joke": [
        "Tiny robot thought: even cloudy days eventually run out of clouds.",
        "My unofficial medical advice is one gentle smile, if available.",
    ],
    "guide_slow_breath": [
        "Let's take one slow breath in, and an even slower breath out.",
        "Breathe in gently, then let the exhale be long and easy.",
    ],
    "suggest_short_pause": [
        "We can pause for a few seconds before doing anything else.",
        "A short pause can help your body settle a little.",
    ],
    "offer_grounding": [
        "Try noticing your feet on the floor and the chair under you.",
        "Pick one steady thing around you and let your attention rest there.",
    ],
    "offer_reassurance": [
        "You're safe with me in this moment.",
        "I'm here, and we can take this slowly.",
    ],
    "offer_check_in": [
        "If you want, we can check what would help most right now.",
        "We can take a quiet second and see what you need next.",
    ],
    "celebrate_with_user": [
        "Let's keep that good feeling around for a bit.",
        "I love hearing that kind of energy from you.",
    ],
    "reinforce_positive_moment": [
        "Moments like this matter and are worth enjoying.",
        "It's nice to let a good moment be a good moment.",
    ],
}


def build_mood_response(trigger, history):
    """Prefer a compact scripted reply; fall back to the LLM if needed."""
    parts = []
    memory = read_memory()
    preferred_name = memory.get("preferred_name")
    scripted_text = trigger.get("scripted_text")
    if scripted_text:
        if preferred_name and random.random() < 0.6:
            scripted_text = f"{preferred_name}, {scripted_text[:1].lower()}{scripted_text[1:]}"
        parts.append(scripted_text)

    for action_name in trigger.get("actions") or []:
        options = ACTION_PHRASES.get(action_name) or []
        if options:
            parts.append(random.choice(options))
            break

    if trigger.get("mood") == "SAD":
        comfort_topics = memory.get("comfort_topics") or []
        favorite_music = memory.get("favorite_music") or []
        if comfort_topics:
            parts.append(
                f"If it helps, we could think about {random.choice(comfort_topics)} for a moment."
            )
        elif favorite_music:
            parts.append(
                f"Maybe a little {random.choice(favorite_music)} could make this moment softer."
            )

    if trigger.get("mood") == "HAPPY":
        favorite_people = memory.get("favorite_people") or []
        if favorite_people:
            parts.append(
                f"This could be a lovely moment to share with {random.choice(favorite_people)} later."
            )

    if trigger.get("streak_count", 0) >= 8:
        parts.append("You've been carrying this feeling for a bit, and I’m still here with you.")

    reply = " ".join(parts).strip()
    if reply:
        return reply

    return get_reply(
        "(No new message from the user — react naturally to how they seem"
        " to be feeling right now, in your own voice.)",
        history,
        extra_system_context=trigger["system_prompt"],
    )


def maybe_react_to_mood(history):
    """
    Poll the latest mood reading from the vision process (via mood_state_io)
    and, if a stable SAD/ANGRY mood has persisted past its cooldown, speak
    a mood-aware reply. No-op if nothing is triggered.
    """
    mood, confidence = read_mood()
    mood_engine.update_mood(mood, confidence)
    handle_mood_escalation()

    trigger = mood_engine.check_trigger(is_speaking=False)
    if not trigger:
        return

    reply = build_mood_response(trigger, history)
    print(f"[mood:{trigger['mood']}] Robot says:", reply)
    history.append(
        {
            "role": "assistant",
            "content": reply,
        }
    )
    set_assistant_output(reply)
    log_event(
        kind="mood_support",
        severity="info",
        message=f"{trigger['label']} support was offered.",
        metadata={
            "mood": trigger["mood"],
            "intent": trigger["intent"],
            "streak_count": trigger.get("streak_count"),
        },
    )
    speak(reply)


def handle_due_reminders(history):
    publish_reminder_state()
    due_reminders = get_due_reminders()
    if not due_reminders:
        return

    for reminder in due_reminders:
        reminder_text = (
            f"Reminder: {reminder['text']}."
            if reminder.get("category") != "sleep"
            else f"Sleep reminder: {reminder['text']}."
        )
        print("[reminder] Robot says:", reminder_text)
        history.append({"role": "assistant", "content": reminder_text})
        set_assistant_output(reminder_text)
        log_event(
            kind="reminder_due",
            severity="warning" if reminder.get("category") == "medicine" else "info",
            message=f"Reminder due: {reminder['text']}.",
            metadata={"id": reminder["id"], "category": reminder.get("category")},
        )
        speak(reminder_text)

    publish_reminder_state()


def handle_mood_escalation():
    escalation = mood_engine.check_escalation()
    if escalation:
        set_alert_state(
            "mood_escalation_active",
            True,
            active_message=(
                f"Persistent {escalation['mood'].lower()} mood detected for "
                f"{escalation['duration_seconds']} seconds."
            ),
            clear_message="Persistent mood escalation cleared.",
            metadata={
                "source": "mood_monitor",
                "mood": escalation["mood"],
                "duration_seconds": escalation["duration_seconds"],
            },
        )
        write_field("mood_escalation_summary", escalation["summary"])
        log_event(
            kind="mood_escalation",
            severity="warning",
            message=escalation["summary"],
            metadata={
                "mood": escalation["mood"],
                "duration_seconds": escalation["duration_seconds"],
            },
        )
        return

    if read_field("mood_escalation_active", max_age_seconds=86400, default=False):
        set_alert_state(
            "mood_escalation_active",
            False,
            active_message="Persistent negative mood detected.",
            clear_message="Persistent mood escalation cleared.",
            metadata={"source": "mood_monitor"},
        )
        write_field("mood_escalation_summary", None)


def conversation_loop(history):
    set_voice_status("active-listening")
    print("Conversation mode. Say 'stop' to go back to sleep.")
    while True:
        set_voice_status("processing")
        handle_due_reminders(history)
        maybe_react_to_mood(history)

        set_voice_status("active-listening")
        record_audio(duration=ACTIVE_LISTEN_SECONDS)
        set_voice_status("transcribing")
        text = transcribe()
        if not text:
            set_voice_status("active-listening")
            continue

        print("You said:", text)
        if has_phrase(text, STOP_PHRASES):
            set_assistant_output("Going back to sleep.")
            set_voice_status("speaking")
            speak("Going back to sleep.")
            set_voice_status("sleep")
            return

        local_answer = answer_local_question(text)
        if local_answer:
            print("Robot says:", local_answer)
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": local_answer})
            set_assistant_output(local_answer)
            set_voice_status("speaking")
            speak(local_answer)
            continue

        set_voice_status("thinking")
        reply = get_reply(text, history)
        print("Robot says:", reply)
        set_assistant_output(reply)
        set_voice_status("speaking")
        speak(reply)


if __name__ == "__main__":
    history = []
    set_voice_status("starting")
    print(
        "Ready. Say 'Wake up' to start, then 'stop' to sleep again. Ctrl+C to"
        " quit."
    )

    try:
        while True:
            listen_for_wake_word()
            conversation_loop(history)
    except KeyboardInterrupt:
        set_voice_status("offline")
        print("\nExiting.")
