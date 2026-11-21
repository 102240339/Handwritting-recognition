"""Small-memory Flask inference app for the trained digit CNN model."""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "model_weights.npz"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Load only the trained arrays
with np.load(WEIGHTS_PATH) as data:
    W1, b1, W2, b2, W3, b3, W4, b4 = [data[f"arr_{i}"].astype(np.float32) for i in range(8)]


def conv_same(x, weights, bias):
    pad = weights.shape[0] // 2
    x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(x, (5, 5), axis=(0, 1))
    windows = windows.transpose(0, 1, 3, 4, 2)
    return np.maximum(np.tensordot(windows, weights, axes=((2, 3, 4), (0, 1, 2))) + bias, 0)


def conv_valid(x, weights, bias):
    windows = np.lib.stride_tricks.sliding_window_view(x, (5, 5), axis=(0, 1))
    windows = windows.transpose(0, 1, 3, 4, 2)
    return np.maximum(np.tensordot(windows, weights, axes=((2, 3, 4), (0, 1, 2))) + bias, 0)


def pool(x):
    h, w, c = x.shape
    return x[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))


def predict_scores(image):
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = pool(conv_same(x[..., None], W1, b1))
    x = pool(conv_valid(x, W2, b2)).reshape(-1)
    x = np.maximum(x @ W3 + b3, 0)
    logits = x @ W4 + b4
    logits -= logits.max()
    scores = np.exp(logits)
    return scores / scores.sum()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")
    if not image_data.startswith("data:image/") or "," not in image_data:
        return jsonify(error="Draw a digit first."), 400
    try:
        raw = base64.b64decode(image_data.split(",", 1)[1], validate=True)
        rgba = Image.open(BytesIO(raw)).convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba).convert("L")
    except Exception:
        return jsonify(error="Could not read the drawing. Please try again."), 400

    ink = Image.fromarray(255 - np.asarray(image, dtype=np.uint8))
    bounds = ink.point(lambda v: 255 if v > 24 else 0).getbbox()
    if not bounds:
        return jsonify(error="Draw a digit first."), 400
    ink = ink.crop(bounds)
    scale = 20.0 / max(ink.width, ink.height)
    digit = ink.resize((max(1, round(ink.width * scale)), max(1, round(ink.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))

    # MNIST digits are centered approximately by their ink center of mass.
    # Centering only the bounding box can leave a long-tailed 9 off-center,
    # making it resemble another class to the model.
    pixels = np.asarray(canvas, dtype=np.float32)
    mass = float(pixels.sum())
    if mass > 0:
        rows, cols = np.indices(pixels.shape)
        center_y = float((rows * pixels).sum() / mass)
        center_x = float((cols * pixels).sum() / mass)
        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))
        canvas = Image.fromarray(pixels.astype(np.uint8)).transform(
            (28, 28), Image.Transform.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
    scores = predict_scores(canvas)
    top = np.argsort(scores)[-2:][::-1]
    return jsonify(digit=int(top[0]), confidence=float(scores[top[0]]),
                   second_digit=int(top[1]), second_confidence=float(scores[top[1]]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", "5000")))
