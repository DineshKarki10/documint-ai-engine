"""Run the DocuMint Gradio app with sensible defaults.

Usage:
    python run.py            # loads .env, launches on http://localhost:7860
    python run.py --port 7860
"""
import os
import sys


def _load_env():
    """Load KEY=VALUE pairs from a local .env file into os.environ."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main():
    _load_env()

    # Keep progress bars quiet during local runs.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Allow the port to be overridden:  python run.py --port 8080
    port = 7860
    if "--port" in sys.argv:
        try:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        except (ValueError, IndexError):
            pass
    os.environ.setdefault("PORT", str(port))

    from app import demo  # noqa: F401  (importing runs the Blocks build)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=__import__("gradio").themes.Soft(primary_hue="indigo", secondary_hue="purple"),
    )


if __name__ == "__main__":
    main()