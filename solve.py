#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from ortools.sat.python import cp_model

def edge_signal(arr: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        diff = np.abs(arr[1:, :, :] - arr[:-1, :, :]).sum(axis=2)
        return diff.sum(axis=1)
    diff = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).sum(axis=2)
    return diff.sum(axis=0)


def best_period(edge: np.ndarray, min_p: int, max_p: int) -> int:
    edge = edge.astype(np.float64)
    edge -= edge.mean()
    best_p = min_p
    best_score = -1e30
    for p in range(min_p, max_p + 1):
        score = (edge[:-p] * edge[p:]).sum()
        if score > best_score:
            best_score = score
            best_p = p
    return best_p


def best_offset(edge: np.ndarray, period: int) -> int:
    best_o = 0
    best_score = -1e30
    for o in range(period):
        score = edge[o::period].sum()
        if score > best_score:
            best_score = score
            best_o = o
    return best_o


def build_lines(length: int, period: int, offset: int) -> List[int]:
    lines = [0]
    pos = offset + 1
    while pos < length:
        lines.append(pos)
        pos += period
    lines.append(length)
    lines = sorted(set(lines))

    if len(lines) < 3:
        return [0, length]

    widths = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
    median = float(np.median(widths))
    if widths[0] < median * 0.5:
        lines = lines[1:]
        widths = widths[1:]
    if widths[-1] < median * 0.5:
        lines = lines[:-1]
    return lines


def detect_grid(arr: np.ndarray) -> Tuple[List[int], List[int]]:
    edge_x = edge_signal(arr, axis=1)
    edge_y = edge_signal(arr, axis=0)

    period_x = best_period(edge_x, 12, 72)
    period_y = best_period(edge_y, 12, 72)

    offset_x = best_offset(edge_x, period_x)
    offset_y = best_offset(edge_y, period_y)

    lines_x = build_lines(arr.shape[1], period_x, offset_x)
    lines_y = build_lines(arr.shape[0], period_y, offset_y)

    return lines_x, lines_y


def cell_patch(arr: np.ndarray, x0: int, x1: int, y0: int, y1: int, margin_ratio: float) -> np.ndarray:
    w = x1 - x0
    h = y1 - y0
    margin = max(1, int(min(w, h) * margin_ratio))
    return arr[y0 + margin : y1 - margin, x0 + margin : x1 - margin, :3]


def classify_cells(arr: np.ndarray, lines_x: List[int], lines_y: List[int]):
    rows = len(lines_y) - 1
    cols = len(lines_x) - 1
    cell_type = np.full((rows, cols), "open", dtype=object)
    horse_cell = None
    cherry_cells = set()
    horse_scores = []

    for r in range(rows):
        for c in range(cols):
            x0, x1 = lines_x[c], lines_x[c + 1]
            y0, y1 = lines_y[r], lines_y[r + 1]
            patch = cell_patch(arr, x0, x1, y0, y1, 0.2)
            cherry_patch = cell_patch(arr, x0, x1, y0, y1, 0.05)
            if patch.size == 0 or cherry_patch.size == 0:
                continue
            avg = patch.mean(axis=(0, 1))
            r_avg, g_avg, b_avg = avg

            blue_like = b_avg > g_avg + 20 and b_avg > r_avg + 20
            if blue_like:
                cell_type[r, c] = "wall"
                continue

            white_mask = (patch[:, :, 0] > 220) & (patch[:, :, 1] > 220) & (patch[:, :, 2] > 220)
            white_count = int(white_mask.sum())
            if white_count > 8:
                horse_scores.append((white_count, r, c))

            red_mask = (cherry_patch[:, :, 0] > 150) & (
                cherry_patch[:, :, 0] > cherry_patch[:, :, 1] + 60
            ) & (cherry_patch[:, :, 0] > cherry_patch[:, :, 2] + 60)
            if red_mask.sum() > 5:
                cherry_cells.add((r, c))

    if horse_scores:
        horse_scores.sort(reverse=True)
        _, hr, hc = horse_scores[0]
        horse_cell = (hr, hc)

    return cell_type, horse_cell, cherry_cells


def solve_exact(cell_type, horse_cell, cherry_cells, max_walls: int):
    if horse_cell is None:
        raise RuntimeError("Horse not detected")

    rows, cols = cell_type.shape
    open_cells = [(r, c) for r in range(rows) for c in range(cols) if cell_type[r, c] != "wall"]
    open_set = set(open_cells)
    idx = {cell: i for i, cell in enumerate(open_cells)}
    n = len(open_cells)

    def is_border(r, c):
        return r == 0 or c == 0 or r == rows - 1 or c == cols - 1

    model = cp_model.CpModel()

    inside = [model.NewBoolVar(f"inside_{i}") for i in range(n)]
    wall = [model.NewBoolVar(f"wall_{i}") for i in range(n)]

    for i in range(n):
        model.Add(inside[i] + wall[i] <= 1)

    horse_idx = idx[horse_cell]
    model.Add(inside[horse_idx] == 1)
    model.Add(wall[horse_idx] == 0)

    for (r, c), i in idx.items():
        if is_border(r, c):
            model.Add(inside[i] == 0)

    for (r, c), i in idx.items():
        for dr, dc in ((1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in open_set:
                continue
            j = idx[(nr, nc)]
            model.Add(inside[i] <= inside[j] + wall[j])
            model.Add(inside[j] <= inside[i] + wall[i])

    model.Add(sum(wall) <= max_walls)

    in_edges: Dict[int, List[cp_model.IntVar]] = {i: [] for i in range(n)}
    out_edges: Dict[int, List[cp_model.IntVar]] = {i: [] for i in range(n)}
    for (r, c), i in idx.items():
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in open_set:
                continue
            j = idx[(nr, nc)]
            f = model.NewIntVar(0, n, f"f_{i}_{j}")
            out_edges[i].append(f)
            in_edges[j].append(f)
            model.Add(f <= n * inside[i])
            model.Add(f <= n * inside[j])

    sum_inside = sum(inside)
    for i in range(n):
        inflow = sum(in_edges[i])
        outflow = sum(out_edges[i])
        if i == horse_idx:
            model.Add(outflow - inflow == sum_inside - 1)
        else:
            model.Add(inflow - outflow == inside[i])

    cherry_list = [idx[c] for c in cherry_cells if c in open_set]
    cherry_count = sum(inside[i] for i in cherry_list)

    solver = cp_model.CpSolver()
    model.Maximize(cherry_count)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found for cherry maximization.")
    max_cherries = solver.Value(cherry_count)

    model.Add(cherry_count == max_cherries)
    model.Maximize(sum_inside)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found for area maximization.")

    walls = {cell for cell, i in idx.items() if solver.Value(wall[i]) == 1}
    region = {cell for cell, i in idx.items() if solver.Value(inside[i]) == 1}
    return walls, region, int(max_cherries)


def draw_solution(img: Image.Image, lines_x, lines_y, walls: set, out_path: str | None = None) -> Image.Image:
    out = img.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    for r, c in walls:
        x0, x1 = lines_x[c], lines_x[c + 1]
        y0, y1 = lines_y[r], lines_y[r + 1]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(255, 0, 0, 160))
    if out_path:
        out.save(out_path)
    return out


def grid_hash(cell_type, horse_cell, cherry_cells, max_walls: int) -> str:
    rows, cols = cell_type.shape
    wall_bits = []
    cherry_bits = []
    for r in range(rows):
        for c in range(cols):
            wall_bits.append("1" if cell_type[r, c] == "wall" else "0")
            cherry_bits.append("1" if (r, c) in cherry_cells else "0")
    parts = [
        f"{rows}x{cols}",
        f"h:{horse_cell[0]},{horse_cell[1]}",
        f"w:{''.join(wall_bits)}",
        f"c:{''.join(cherry_bits)}",
        f"m:{max_walls}",
    ]
    digest = hashlib.sha256("|".join(parts).encode("ascii")).hexdigest()
    return digest


def solve_image_cached(img: Image.Image, max_walls: int, cache_dir: Path):
    arr = np.array(img.convert("RGB"), dtype=np.int16)
    lines_x, lines_y = detect_grid(arr)
    cell_type, horse_cell, cherry_cells = classify_cells(arr, lines_x, lines_y)

    if horse_cell is None:
        raise RuntimeError("Horse not detected. Try adjusting thresholds.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = grid_hash(cell_type, horse_cell, cherry_cells, max_walls)
    out_path = cache_dir / f"{signature}.png"
    stats_path = cache_dir / f"{signature}.json"

    stats = {
        "grid_rows": len(lines_y) - 1,
        "grid_cols": len(lines_x) - 1,
        "horse_cell": horse_cell,
        "cherries_detected": len(cherry_cells),
        "cherries_enclosed": None,
        "walls_used": None,
    }
    if out_path.exists() and stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            cached_stats = json.load(handle)
        if isinstance(cached_stats.get("horse_cell"), list):
            cached_stats["horse_cell"] = tuple(cached_stats["horse_cell"])
        stats.update(cached_stats)
        return out_path, stats

    walls, region, max_cherries = solve_exact(cell_type, horse_cell, cherry_cells, max_walls)
    draw_solution(img, lines_x, lines_y, walls, str(out_path))
    stats["cherries_enclosed"] = max_cherries
    stats["walls_used"] = len(walls)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=True)
    return out_path, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Enclose the horse with minimal walls.")
    parser.add_argument("input", nargs="?", default="img.png")
    parser.add_argument("output", nargs="?", default="solution.png")
    parser.add_argument("--max-walls", type=int, default=11)
    args = parser.parse_args()

    img = Image.open(args.input)
    out_path, stats = solve_image_cached(img, args.max_walls, Path("."))
    if out_path != Path(args.output):
        out = Image.open(out_path)
        out.save(args.output)

    print(f"Grid: {stats['grid_rows']} rows x {stats['grid_cols']} cols")
    print(f"Horse cell: {stats['horse_cell']}")
    print(f"Cherries detected: {stats['cherries_detected']}")
    print(f"Cherries enclosed: {stats['cherries_enclosed']}")
    print(f"Walls used: {stats['walls_used']}")


if __name__ == "__main__":
    main()
