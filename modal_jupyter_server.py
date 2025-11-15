# Start a jupyter kernel on Modal as a sandbox with persistent storage
# See: https://modal.com/docs/examples/jupyter_sandbox
#
# Right now the image is composed of necessary dependencies for running Whisper models
# via FasterWhisper and huggingface transformers. Add other libraries to the kernel as needed
# or install them from the running notebook.
#
# run with:
# modal run start_jupyter_kernel.py
#
# then see sandbox dashboard:
# https://modal.com/sandboxes/personalizedmodels/main

###########################
# Adjust these
#
JUPYTER_PORT = 8888
TIMEOUT = 6000 # seconds
GPU_TYPE = 'T4' # choose according to: https://modal.com/pricing
MOUNT_VOLUME = "pytorch-book"
PERSISTENCE_FOLDER = "/home/gson"
###########################

import json
import secrets
import time
import urllib.request
import signal
import sys

import modal

app = modal.App.lookup("jupyter_kernel", create_if_missing=True)

# Create or reference a persistent volume
volume = modal.Volume.from_name(MOUNT_VOLUME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install(
        "wget",
        "git",
        "libsndfile1",
        "libsndfile1-dev",
        "ffmpeg",
        "pkg-config",
	    "build-essential",
        "graphviz",
        "wget")
    .pip_install(
        "jupyter~=1.1.0",
        "numpy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "librosa",
        "soundfile",
        "audioread",
        "datasets[audio]==3.6.0", # use 3.6.0 as latest version (4.0.0) has breaking changes
        "matplotlib",
        "evaluate",
        "huggingface_hub",
        "torch",
        "torchaudio",
        "torchvision",
        "torchviz",
        "graphviz",
        "tensorboard",
        "ctranslate2",
        "faster_whisper",
        "transformers",
    )
    # Create the pytorch-book directory in the image
    .run_commands(f"mkdir -p {PERSISTENCE_FOLDER}")
)

token = secrets.token_urlsafe(13)
token_secret = modal.Secret.from_dict({"JUPYTER_TOKEN": token})

print("🏖️  Creating sandbox with persistent storage")

def graceful_shutdown():
    """Handle graceful shutdown of the sandbox"""
    print("\n🛑  Shutting down sandbox...")
    print("🔄  Modal automatically commits volume changes on container shutdown")
    try:
        if 'sandbox' in globals():
            sandbox.terminate()
            print("✅  Sandbox terminated gracefully")
    except Exception as e:
        print(f"⚠️  Note: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    graceful_shutdown()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        "jupyter",
        "notebook",
        "--no-browser",
        "--allow-root",
        "--ip=0.0.0.0",
        f"--port={JUPYTER_PORT}",
        "--NotebookApp.allow_origin='*'",
        "--NotebookApp.allow_remote_access=1",
        "--NotebookApp.log_level=DEBUG",
        encrypted_ports=[JUPYTER_PORT],
        secrets=[token_secret],
        timeout=TIMEOUT,
        image=image,
        app=app,
        gpu=GPU_TYPE,
        # Mount the persistent volume
        volumes={PERSISTENCE_FOLDER: volume},
    )

print(f"🏖️  Sandbox ID: {sandbox.object_id}")
print(f"🏖️  Persistent storage mounted at: {PERSISTENCE_FOLDER}")

tunnel = sandbox.tunnels()[JUPYTER_PORT]
url = f"{tunnel.url}/?token={token}"
print(f"🏖️  Jupyter notebook is running at: {url}")

def is_jupyter_up():
    try:
        response = urllib.request.urlopen(f"{tunnel.url}/api/status?token={token}")
        if response.getcode() == 200:
            data = json.loads(response.read().decode())
            return data.get("started", False)
    except Exception:
        return False
    return False

# Wait for Jupyter to start
startup_timeout = 60  # seconds
start_time = time.time()
while time.time() - start_time < startup_timeout:
    if is_jupyter_up():
        print("🏖️  Jupyter is up and running!")
        print(f"🏖️  Your persistent files are available at {PERSISTENCE_FOLDER}")
        print()
        print("🏖️  Press Ctrl+C to stop the sandbox")
        
        # Keep the script running
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)
        
        break
    time.sleep(1)
else:
    print("🏖️  Timed out waiting for Jupyter to start.")
    graceful_shutdown()
