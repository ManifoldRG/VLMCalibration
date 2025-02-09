#!/usr/bin/env python3
import subprocess
import time
import sys
import os


def run_command(command: str) -> tuple[int, str, str]:
    """
    Execute a shell command and return its exit code, stdout, and stderr
    """
    try:
        process = subprocess.Popen(
            command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        return (process.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"))
    except Exception as e:
        return 1, "", str(e)


def check_nvidia_smi() -> bool:
    """Check if nvidia-smi is working"""
    code, out, err = run_command("nvidia-smi")
    return code == 0


def monitor_nvidia_smi():
    """Monitor nvidia-smi output continuously"""
    try:
        while True:
            # Clear screen and move cursor to top
            print("\033[2J\033[H", end="")  # Clear screen and move to (0,0)
            code, out, err = run_command("nvidia-smi")
            if code == 0:
                print(out, end="")  # end="" prevents additional newline
            else:
                print(f"Error: {err}", end="")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
        sys.exit(0)


def main():
    print("Starting NVIDIA GPU monitoring (Ctrl+C to stop)...")
    if not check_nvidia_smi():
        print("Error: nvidia-smi is not working!")
        sys.exit(1)

    monitor_nvidia_smi()


if __name__ == "__main__":
    main()
