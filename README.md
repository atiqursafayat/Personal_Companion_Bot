# Pesonal_Companion_Bot

## How to start-
```
python3 -m venv venv
source venv/bin/activate
pip install groq numpy pygame sounddevice soundfile piper-tts
mkdir -p piper_voices
cd piper_voices/
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ..
export GROQ_API_KEY="your_api_key_here"

pip install ultralytics ncnn
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='ncnn', imgsz=320)"

```
