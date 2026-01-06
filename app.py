#!/usr/bin/env python3
import base64
import io
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image

from solve import solve_image

APP_ROOT = Path(__file__).parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/solve")
def solve():
    file = request.files.get("image")
    if not file:
        return render_template("index.html", error="Please upload an image.")

    try:
        max_walls = int(request.form.get("max_walls", "11"))
    except ValueError:
        max_walls = 11

    if max_walls <= 0:
        max_walls = 1

    img = Image.open(file.stream)
    out, stats = solve_image(img, max_walls)

    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return render_template(
        "result.html",
        image_data=encoded,
        max_walls=max_walls,
        stats=stats,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
