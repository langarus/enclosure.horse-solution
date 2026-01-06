#!/usr/bin/env python3
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image

from solve import solve_image_cached

APP_ROOT = Path(__file__).parent
CACHE_DIR = APP_ROOT / "static" / "cache"

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
    out_path, stats = solve_image_cached(img, max_walls, CACHE_DIR)
    image_url = f"/static/cache/{out_path.name}"

    return render_template(
        "result.html",
        image_url=image_url,
        max_walls=max_walls,
        stats=stats,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
