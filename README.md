# Personal Companion Bot

Run `target_mood_tracker.py` and `voice_assistant.py` together. The tracker owns
the default camera; the assistant uses the default microphone.

An audio output device is optional. If no default speaker is available, the
assistant continues in text-only mode and prints its replies. To explicitly
disable spoken output, start it with `COMPANION_TEXT_ONLY=1`.

## Setup

Create and activate the virtual environment, then install the voice-assistant
dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install groq numpy pygame sounddevice soundfile piper-tts
```

Download the Piper voice model:

```bash
mkdir -p piper_voices
cd piper_voices/
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ..
```

Set the Groq API key for the current terminal session:

```bash
export GROQ_API_KEY="your_api_key_here"
```

Install the vision dependencies and export the YOLO model to NCNN:

```bash
pip install ultralytics ncnn
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='ncnn', imgsz=320)"
```

## How to start

Start both the mood tracker and voice assistant together with `main.py`:

```bash
source venv/bin/activate
export GROQ_API_KEY="your_api_key_here"
python3 main.py
```

Press `Ctrl+C` in the terminal to stop both programs. Pressing `q` in the mood
tracker window also causes `main.py` to stop the voice assistant.

You can still run the processes separately for debugging. Open two terminals in
the project directory and use the following commands.

Terminal 1 — mood tracker:

```bash
source venv/bin/activate
python3 target_mood_tracker.py
```

Terminal 2 — voice assistant:

```bash
source venv/bin/activate
export GROQ_API_KEY="your_api_key_here"
python3 voice_assistant.py
```

To run without a speaker or other output device:

```bash
source venv/bin/activate
export GROQ_API_KEY="your_api_key_here"
COMPANION_TEXT_ONLY=1 python3 voice_assistant.py
```

Press `q` in the mood-tracker window to stop it. Press `Ctrl+C` in the voice
assistant terminal to stop the assistant.

## Device voice commands

- `Set volume to 40 percent`, `volume up`, `mute`, `volume level`
- `Set brightness to 50 percent`, `brightness up`, `brightness level`
- `Battery status`, `Wi-Fi status`, `turn Wi-Fi on/off`
- `Turn Bluetooth on/off`, `Bluetooth status`
- `Open calculator`, `open browser`, `open google.com`
- `Take a screenshot`, `take a picture`, `lock the screen`
- `Play`, `pause`, `next song`, `previous song`
- `CPU temperature`, `disk space`
- `Shut down the computer`, `restart the computer`
- `Delete the latest picture`

Bluetooth off, Wi-Fi off, screen locking, shutdown, restart, deleting a picture,
and clearing reminders require a spoken `yes` confirmation. Say `no` to cancel.
Confirmations expire after 20 seconds.

Linux integrations use `wpctl`, `brightnessctl`, `bluetoothctl`, `nmcli`,
`playerctl`, `loginctl`, and `xdg-open`. Screenshots use the first installed tool
from `gnome-screenshot`, `spectacle`, `scrot`, or `grim`.

## Timers and reminders

- `Set a timer for ten minutes`
- `Remind me to take medicine at 8 PM`
- `Remind me to stretch in thirty minutes`
- `List my reminders`
- `Cancel all reminders`

Timers and reminders are stored in `.companion_data/reminders.json`, survive
assistant restarts, and are announced in both sleep and conversation modes.
Pictures and screenshots are stored in `captures/`.
