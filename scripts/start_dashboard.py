"""
scripts/start_dashboard.py — Launches API + Streamlit Dashboard

Starts both the FastAPI backend and Streamlit frontend concurrently.
FastAPI runs on port 8000 (background process).
Streamlit runs on port 8501 (foreground).

Usage:
    python scripts/start_dashboard.py
"""

import subprocess
import sys
import os
import time
import signal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


def main():
    print("=" * 50)
    print("  GigScore — Starting Services")
    print("=" * 50)

    processes = []

    try:
        # Start FastAPI backend
        print("\n🚀 Starting FastAPI backend on port 8000...")
        api_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app",
             "--host", "0.0.0.0", "--port", "8000", "--reload"],
            cwd=PROJECT_ROOT,
        )
        processes.append(api_process)
        time.sleep(3)  # Wait for API to initialize

        print("✅ FastAPI running at: http://localhost:8000")
        print("📚 API Docs at: http://localhost:8000/docs")

        # Start Streamlit dashboard
        print("\n🎨 Starting Streamlit dashboard on port 8501...")
        streamlit_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
             "--server.port", "8501",
             "--server.headless", "true",
             "--theme.base", "dark",
             "--theme.primaryColor", "#6366f1",
             "--theme.backgroundColor", "#0f172a",
             "--theme.secondaryBackgroundColor", "#1e293b",
             "--theme.textColor", "#e2e8f0"],
            cwd=PROJECT_ROOT,
        )
        processes.append(streamlit_process)

        print("✅ Dashboard running at: http://localhost:8501")
        print("\n" + "=" * 50)
        print("  Press Ctrl+C to stop all services")
        print("=" * 50)

        # Wait for processes
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        print("✅ All services stopped")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        for p in processes:
            p.terminate()
        sys.exit(1)


if __name__ == '__main__':
    main()
