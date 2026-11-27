"""Hugging Face Spaces entrypoint.

Spaces look for `app.py` and a module-level `demo`. Deploying still needs a
Hugging Face account; this file is what you point the Space at.
"""

from __future__ import annotations

import os

from qcdemo.space_app import build_upload_demo

demo = build_upload_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
