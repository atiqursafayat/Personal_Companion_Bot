import multiprocessing
import os
import signal
import sys
import time

def run_vision_subsystem():
    """
    Launches target_mood_tracker.py execution loop.
    Encapsulated inside a process wrapper.
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

def main():
    print("==================================================")
    print("      STARTING MULTIMODAL COMPANION ROBOT       ")
    print("==================================================")
    
    # Instantiate child processes for Vision and Voice loops
    vision_process = multiprocessing.Process(
        target=run_vision_subsystem, name="VisionSubsystem"
    )
    voice_process = multiprocessing.Process(
        target=run_voice_subsystem, name="VoiceSubsystem"
    )

    # Set as non-daemon so processes complete cleanly on termination
    vision_process.daemon = False
    voice_process.daemon = False

    print("[MAIN] Spawning Vision Subsystem...")
    vision_process.start()

    print("[MAIN] Spawning Voice Subsystem...")
    voice_process.start()

    def shutdown_handler(signum, frame):
        print("\n[MAIN] Shutdown signal received. Terminating processes...")
        
        if voice_process.is_alive():
            print("[MAIN] Stopping Voice Subsystem...")
            voice_process.terminate()
            voice_process.join(timeout=2)

        if vision_process.is_alive():
            print("[MAIN] Stopping Vision Subsystem...")
            vision_process.terminate()
            vision_process.join(timeout=2)

        print("[MAIN] System shutdown complete.")
        sys.exit(0)

    # Register OS signals for clean exit (Ctrl+C / SIGTERM)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print("[MAIN] Both processes operational. Press Ctrl+C to terminate.")

    # Main thread monitoring loop
    while True:
        time.sleep(1)
        # Verify both worker processes remain alive
        if not vision_process.is_alive():
            print("[WARNING] Vision subsystem stopped unexpectedly.")
            break
        if not voice_process.is_alive():
            print("[WARNING] Voice subsystem stopped unexpectedly.")
            break

if __name__ == "__main__":
    main()
