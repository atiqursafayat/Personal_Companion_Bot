# Personal Companion Bot

Three cooperating processes managed by `main.py`:

- **Vision** (`target_mood_fixed.py`) — owns the default camera, detects people and
  reads emotions via YOLO + DeepFace, writes the current mood to a shared state file.
- **Voice** (`voice_assistant.py`) — listens on the default microphone, answers local
  commands, and falls back to a Groq LLM for open conversation. Reacts to the mood
  detected by the vision process.
- **Dashboard** (`dashboard/server.py`) — serves a real-time web dashboard at
  `http://localhost:8080` showing the live camera feed, current mood, system status,
  reminders, and recent captures.

An audio output device is optional. If no default speaker is available, the
assistant continues in text-only mode and prints its replies. To explicitly
disable spoken output, start it with `COMPANION_TEXT_ONLY=1`.

## Setup

### 1 — System packages

Install the PortAudio library required by `sounddevice`:

```bash
sudo apt install -y libportaudio2 portaudio19-dev
```

Optional — needed only for brightness and media voice commands:

```bash
sudo apt install -y brightnessctl playerctl
```

### 2 — Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3 — Piper voice model

```bash
mkdir -p piper_voices
cd piper_voices/
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ..
```

### 4 — YOLO NCNN model

```bash
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='ncnn', imgsz=320)"
```

This downloads `yolov8n.pt` on the first run and creates the `yolov8n_ncnn_model/`
directory used by the vision process.

### 5 — Groq API key

Get a free key at [console.groq.com](https://console.groq.com) → API Keys, then
export it in every terminal session where you run the bot:

```bash
export GROQ_API_KEY="your_api_key_here"
```

## How to start

Start all three processes together:

```bash
source venv/bin/activate
export GROQ_API_KEY="your_api_key_here"
python3 main.py
```

Then open **http://localhost:8080** in your browser to see the live dashboard.

Press `Ctrl+C` in the terminal to stop all processes. Pressing `q` in the
vision window also causes `main.py` to stop the other processes.

## Running processes separately (debugging)

Open three terminals in the project directory.

**Terminal 1 — vision:**

```bash
source venv/bin/activate
python3 target_mood_fixed.py
```

**Terminal 2 — voice assistant:**

```bash
source venv/bin/activate
export GROQ_API_KEY="your_api_key_here"
python3 voice_assistant.py
```

**Terminal 3 — dashboard only:**

```bash
source venv/bin/activate
python3 dashboard/server.py
```

To run the voice assistant without a speaker:

```bash
source venv/bin/activate
export GROQ_API_KEY="your_api_key_here"
COMPANION_TEXT_ONLY=1 python3 voice_assistant.py
```

## Dashboard

The dashboard is served at `http://localhost:8080` (LAN-accessible via your machine's
IP on the same port). It shows:

- **Live camera feed** — annotated MJPEG stream from the vision process
- **Current mood** — detected emotion and confidence percentage, updated every second
- **System status** — battery, CPU temperature, disk usage, Wi-Fi network
- **Upcoming reminders** — all pending timers and reminders with their due times
- **Recent captures** — thumbnail gallery of saved photos and screenshots

The dashboard updates automatically via Server-Sent Events — no page reload needed.

### Dashboard environment variables

| Variable | Default | Description |
|---|---|---|
| `DASHBOARD_HOST` | `0.0.0.0` | Bind address (`127.0.0.1` for localhost-only) |
| `DASHBOARD_PORT` | `8080` | Port the dashboard listens on |
| `LATEST_FRAME_PATH` | `/tmp/robot_latest_frame.jpg` | Shared camera frame path |

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
