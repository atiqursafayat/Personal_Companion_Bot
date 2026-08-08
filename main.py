"""Launch and supervise the mood tracker and voice assistant together."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parent
PROGRAMS = ("target_mood_fixed.py", "voice_assistant.py")
SHUTDOWN_TIMEOUT_SECONDS = 5


def start_program(script_name):
    script_path = PROJECT_DIRECTORY / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Required program not found: {script_path}")

    print(f"[main] Starting {script_name}...", flush=True)
    return subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIRECTORY,
        env=os.environ.copy(),
        start_new_session=True,
    )


def stop_programs(processes):
    running = [process for process in processes if process.poll() is None]
    for process in running:
        process.terminate()

    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
    for process in running:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass

    for process in running:
        if process.poll() is None:
            print(f"[main] Force-stopping process {process.pid}...", flush=True)
            process.kill()
            process.wait()


def main():
    processes = []
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)

    try:
        for script_name in PROGRAMS:
            processes.append(start_program(script_name))

        print(
            "[main] Mood tracker and voice assistant are running. "
            "Press Ctrl+C to stop both.",
            flush=True,
        )

        while not stop_requested:
            for script_name, process in zip(PROGRAMS, processes):
                return_code = process.poll()
                if return_code is not None:
                    print(
                        f"[main] {script_name} exited with code {return_code}; "
                        "stopping the other program.",
                        flush=True,
                    )
                    return return_code if return_code != 0 else 0
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[main] Shutdown requested.", flush=True)
    finally:
        stop_programs(processes)
        signal.signal(signal.SIGTERM, previous_sigterm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
