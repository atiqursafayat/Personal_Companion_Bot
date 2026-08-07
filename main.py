import multiprocessing
import os
import signal
import socket
import sys
import time

DASHBOARD_PORT = 8080


def run_vision_subsystem():
    """
    Launches target_mood_tracker.py execution loop.
    Encapsulated inside a process wrapper.

    NOTE: this is the only subsystem allowed to open the physical camera
    (cv2.VideoCapture). It publishes annotated frames via frame_io.py for
    the dashboard to stream, instead of opening its own cv2 window.
    """
    try:
        # Import target_mood_tracker dynamically within the process context
        # to ensure independent memory and GIL execution space.
        import target_mood_tracker
    except Exception as e:
        print(f"[ERROR] Vision process crashed: {e}")

def run_voice_subsystem():
    """
    Launches voice_assistant.py execution loop.
    Encapsulated inside a process wrapper.
    """
    try:
        # Import voice assistant module components
        import voice_assistant

        history = []
        print("[VOICE] System ready. Say 'Wake up' to activate.")

        while True:
            voice_assistant.listen_for_wake_word()
            voice_assistant.conversation_loop(history)
    except Exception as e:
        print(f"[ERROR] Voice process crashed: {e}")


def run_dashboard_subsystem():
    """
    Launches the Flask web dashboard (camera feed + live status panel).
    This replaces the old cv2.imshow debug window -- instead of a local
    window, the camera feed and robot state are viewable from any browser
    on the same network at http://<pi-ip>:8080.
    """
    try:
        import dashboard_server

        # use_reloader=False is important: Flask's dev-server reloader
        # spawns a child process of its own, which would fight with
        # multiprocessing's own process management here.
        dashboard_server.app.run(
            host="0.0.0.0", port=DASHBOARD_PORT, threaded=True,
            debug=False, use_reloader=False,
        )
    except Exception as e:
        print(f"[ERROR] Dashboard process crashed: {e}")


def get_local_ip():
    """Best-effort LAN IP so we can print a clickable dashboard link."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    print("==================================================")
    print("      STARTING MULTIMODAL COMPANION ROBOT       ")
    print("==================================================")
    
    # Instantiate child processes for Vision, Voice, and Dashboard loops
    vision_process = multiprocessing.Process(
        target=run_vision_subsystem, name="VisionSubsystem"
    )
    voice_process = multiprocessing.Process(
        target=run_voice_subsystem, name="VoiceSubsystem"
    )
    dashboard_process = multiprocessing.Process(
        target=run_dashboard_subsystem, name="DashboardSubsystem"
    )

    # Set as non-daemon so processes complete cleanly on termination
    vision_process.daemon = False
    voice_process.daemon = False
    dashboard_process.daemon = False

    print("[MAIN] Spawning Vision Subsystem...")
    vision_process.start()

    print("[MAIN] Spawning Voice Subsystem...")
    voice_process.start()

    print("[MAIN] Spawning Dashboard Subsystem...")
    dashboard_process.start()

    def shutdown_handler(signum, frame):
        print("\n[MAIN] Shutdown signal received. Terminating processes...")

        if voice_process.is_alive():
            print("[MAIN] Stopping Voice Subsystem...")
            voice_process.terminate()
            voice_process.join(timeout=2)

        if dashboard_process.is_alive():
            print("[MAIN] Stopping Dashboard Subsystem...")
            dashboard_process.terminate()
            dashboard_process.join(timeout=2)

        if vision_process.is_alive():
            print("[MAIN] Stopping Vision Subsystem...")
            vision_process.terminate()
            vision_process.join(timeout=2)

        print("[MAIN] System shutdown complete.")
        sys.exit(0)

    # Register OS signals for clean exit (Ctrl+C / SIGTERM)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print("[MAIN] All subsystems operational. Press Ctrl+C to terminate.")

    # Give the dashboard's Flask server a moment to bind before printing
    # the link, so the URL is actually live by the time it's shown.
    time.sleep(1.5)
    dashboard_url = f"http://{get_local_ip()}:{DASHBOARD_PORT}"
    print("--------------------------------------------------")
    print(f"[DASHBOARD] Live camera feed + status: {dashboard_url}")
    print("--------------------------------------------------")

    # Main thread monitoring loop
    while True:
        time.sleep(1)
        # Verify all worker processes remain alive
        if not vision_process.is_alive():
            print("[WARNING] Vision subsystem stopped unexpectedly.")
            break
        if not voice_process.is_alive():
            print("[WARNING] Voice subsystem stopped unexpectedly.")
            break
        if not dashboard_process.is_alive():
            print("[WARNING] Dashboard subsystem stopped unexpectedly.")
            break

if __name__ == "__main__":
    main()
