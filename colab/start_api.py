"""Start FastAPI in the current Colab kernel so it shares the loaded model."""
import json
import os
import socket
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

BACKEND_DIR = Path("/content/t2v_backend")
assert (BACKEND_DIR / "server.py").exists(), "Re-run Cell 3. Missing /content/t2v_backend/server.py"
assert (BACKEND_DIR / "generator.py").exists(), "Re-run Cell 3. Missing /content/t2v_backend/generator.py"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
print("Backend files ready:", BACKEND_DIR)

os.environ["T2V_OUTPUT_DIR"] = "/content/outputs"
os.environ["T2V_SKIP_PRELOAD"] = "1"
Path("/content/outputs").mkdir(parents=True, exist_ok=True)

FORCE_RESTART = bool(globals().get("FORCE_RESTART", False))


def port_open(port=8000):
    s = socket.socket()
    s.settimeout(0.4)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def local_health():
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode())
    except Exception:
        return None


def stop_old():
    old = globals().get("T2V_UVICORN")
    if old is None:
        return
    print("Stopping previous uvicorn")
    try:
        old.should_exit = True
        old.force_exit = True
    except Exception as exc:
        print("stop warning:", exc)
    time.sleep(1.5)


healthy = local_health()
if healthy and healthy.get("ok") and not FORCE_RESTART:
    print("FastAPI already running in this process. Not starting another server.")
    print("FastAPI is healthy on http://127.0.0.1:8000")
    print(healthy)
else:
    if FORCE_RESTART:
        stop_old()
    healthy = None if FORCE_RESTART else local_health()
    if healthy and healthy.get("ok"):
        print("FastAPI already running in this process. Not starting another server.")
        print("FastAPI is healthy on http://127.0.0.1:8000")
        print(healthy)
    else:
        import generator  # same module Cell 4 loaded; do not reload
        import server
        from server import app
        import uvicorn

        print("generator loaded flag:", generator.model_info().get("loaded"))
        boot_error = []
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", lifespan="off")
        uv_server = uvicorn.Server(config)
        uv_server.install_signal_handlers = False

        def run_api():
            try:
                uv_server.run()
            except Exception:
                boot_error.append(traceback.format_exc())
                traceback.print_exc()

        threading.Thread(target=run_api, daemon=True, name="t2v-fastapi").start()
        globals()["T2V_UVICORN"] = uv_server

        healthy = None
        for _ in range(40):
            time.sleep(0.25)
            if boot_error:
                break
            healthy = local_health()
            if healthy and healthy.get("ok"):
                break

        if not (healthy and healthy.get("ok")):
            print("FastAPI failed to start")
            if boot_error:
                print(boot_error[-1])
            elif port_open(8000):
                print("Port 8000 is open but /health did not return ok.")
            else:
                print("Nothing is listening on 127.0.0.1:8000")
            raise SystemExit("FastAPI failed to start")
        print("FastAPI is healthy on http://127.0.0.1:8000")
        print(healthy)
