# Hugging Face Spaces

This repository is set up so a Gradio Space can run the upload demo. I did not
deploy it. Creating a Space needs a Hugging Face account, and this environment
does not have one.

What the Space uses:

- `app.py` at the repository root (`demo = build_upload_demo()`)
- `requirements.txt` (CPU PyTorch index included)
- the `defectclip/` and `qcdemo/` packages

The Space does not download MVTec. The first score loads OpenAI CLIP ViT-B-16,
which takes a short while and needs outbound network on the Space.

## Create the Space

1. On huggingface.co, create a new Space, SDK **Gradio**.
2. Either push this repo to the Space, or create the Space from the GitHub
   repo if your account is allowed to.
3. Put this block at the top of the Space README (Hugging Face reads it; the
   GitHub README deliberately does not start with it):

```yaml
---
title: CLIP Defect Check
emoji: 🔍
colorFrom: blue
colorTo: slate
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
---
```

4. The Space hardware can stay on CPU. The model card should say the score is
   an uncalibrated cosine gap, same as the API note.

`python app.py` locally is the same UI the Space serves, on port 7860
(or `$PORT`).
