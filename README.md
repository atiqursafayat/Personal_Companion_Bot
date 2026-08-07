# Personal Companion Bot

## Dashboard

Activate the project environment and start the dashboard:

```bash
source venv/bin/activate
python dashboard_server.py
```

Open `http://localhost:8080` locally, or use the Raspberry Pi's LAN IP from
another device. The full supervisor in `main.py` also starts the dashboard.

Available routes:

- `/` — responsive care dashboard
- `/api/state` — current robot and sensor state as JSON
- `/api/health` — lightweight server health check
- `/video_feed` — MJPEG camera stream
