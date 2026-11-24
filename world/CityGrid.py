import math
from functools import cache

import numpy as np

from world.CityMap import loadFeatures


def insideMask(X, Y, pts):
    """Even-odd point-in-polygon test for flat arrays of cell centres."""
    inside = np.zeros(X.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1], strict=True):
        if y1 == y2:
            continue
        crosses = (y1 > Y) != (y2 > Y)
        xCross = (x2 - x1) * (Y - y1) / (y2 - y1) + x1
        inside ^= crosses & (X < xCross)
    return inside


def markLine(grid, pts, value, width, height, cellW, cellH):
    """Mark every cell that a polyline passes through (sampled at a third of a cell)."""
    rows, cols = grid.shape
    step = min(cellW, cellH) / 3
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:], strict=True):
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / step) + 1)
        t = np.linspace(0.0, 1.0, n)
        xs = x1 + t * (x2 - x1)
        ys = y1 + t * (y2 - y1)
        cs = np.clip((xs / cellW).astype(int), 0, cols - 1)
        rs = np.clip(((height - ys) / cellH).astype(int), 0, rows - 1)
        grid[rs, cs] = value


@cache
def rasterise(rawPath, rows: int, cols: int) -> np.ndarray:
    """Terrain grid, row 0 = north, col 0 = west. Layers in order: mud, grass, water, roads, buildings."""
    features = loadFeatures(rawPath)
    width, height = features["size"]
    cellW, cellH = width / cols, height / rows
    centreX = (np.arange(cols) + 0.5) * cellW
    centreY = height - (np.arange(rows) + 0.5) * cellH
    X, Y = np.meshgrid(centreX, centreY)
    X, Y = X.ravel(), Y.ravel()
    grid = np.full((rows, cols), 2, dtype=int)  # mud: open or unpaved land
    for g in features["green"]:
        grid[insideMask(X, Y, g["pts"]).reshape(rows, cols)] = 1
    for w in features["water"]:
        if w["closed"]:
            grid[insideMask(X, Y, w["pts"]).reshape(rows, cols)] = 3
        else:
            markLine(grid, w["pts"], 3, width, height, cellW, cellH)
    for r in features["roads"]:
        if r["cls"] != "footway":
            markLine(grid, r["pts"], 0, width, height, cellW, cellH)
    for b in features["buildings"]:
        grid[insideMask(X, Y, b["pts"]).reshape(rows, cols)] = -1
    return grid


def nearestFree(grid, cell, taken):
    free = [tuple(int(v) for v in c) for c in np.argwhere(grid >= 0)]
    free = [c for c in free if c not in taken]
    return min(free, key=lambda c: abs(c[0] - cell[0]) + abs(c[1] - cell[1]))


def cityGrid(rawPath, start, stops, rows=39, cols=39):
    """Real-city demo terrain plus start and stops snapped to distinct nearest free cells.
    Returns (grid, start, stops)."""
    grid = rasterise(rawPath, rows, cols)
    snapped = []
    for cell in [start, *stops]:
        cell = tuple(cell)
        snapped.append(cell if grid[cell] >= 0 and cell not in snapped else nearestFree(grid, cell, snapped))
    return grid, snapped[0], tuple(snapped[1:])
