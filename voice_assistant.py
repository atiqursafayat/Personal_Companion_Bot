import os
import shutil
import subprocess
import sys
import time

from groq import Groq
import pygame
import sounddevice as sd
import soundfile as sf

from voice_commands import answer_local_question
from mood_reactions import MoodReactionEngine
from mood_state_io import read_mood

# Initialize pygame mixer once at startup
pygame.mixer.init()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
VOICE_MODEL = (
    "piper_voices/en_US-lessac-medium.onnx"
)
WAKE_PHRASES = ("wake up",)
STOP_PHRASES = ("stop",)
SLEEP_LISTEN_SECONDS = 3
ACTIVE_LISTEN_SECONDS = 5
PREFERRED_SAMPLE_RATES = (16000, 22050, 44100, 48000)


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
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.01)


def listen_for_wake_word():
    print("Sleep mode. Say 'Wake up' to activate.")
    while True:
        record_audio(duration=SLEEP_LISTEN_SECONDS, filename="sleep.wav")
        text = transcribe("sleep.wav")
        if not text:
            continue

        print("Heard:", text)
        if has_phrase(text, WAKE_PHRASES):
            speak("I'm awake.")
            return


mood_engine = MoodReactionEngine()


def maybe_react_to_mood(history):
    """
    Poll the latest mood reading from the vision process (via mood_state_io)
    and, if a stable SAD/ANGRY mood has persisted past its cooldown, speak
    a mood-aware reply. No-op if nothing is triggered.
    """
    mood, _confidence = read_mood()
    mood_engine.update_mood(mood)

    trigger = mood_engine.check_trigger(is_speaking=False)
    if not trigger:
        return

    reply = get_reply(
        "(No new message from the user — react naturally to how they seem"
        " to be feeling right now, in your own voice.)",
        history,
        extra_system_context=trigger["system_prompt"],
    )
    print(f"[mood:{trigger['mood']}] Robot says:", reply)
    speak(reply)


def conversation_loop(history):
    print("Conversation mode. Say 'stop' to go back to sleep.")
    while True:
        maybe_react_to_mood(history)

        record_audio(duration=ACTIVE_LISTEN_SECONDS)
        text = transcribe()
        if not text:
            continue

        print("You said:", text)
        if has_phrase(text, STOP_PHRASES):
            speak("Going back to sleep.")
            return

        local_answer = answer_local_question(text)
        if local_answer:
            print("Robot says:", local_answer)
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": local_answer})
            speak(local_answer)
            continue

        reply = get_reply(text, history)
        print("Robot says:", reply)
        speak(reply)


if __name__ == "__main__":
    history = []
    print(
        "Ready. Say 'Wake up' to start, then 'stop' to sleep again. Ctrl+C to"
        " quit."
    )

    try:
        while True:
            listen_for_wake_word()
            conversation_loop(history)
    except KeyboardInterrupt:
        print("\nExiting.")
