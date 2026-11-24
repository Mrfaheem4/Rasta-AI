import json
import math
import random
from pathlib import Path

from core.Config import cityBounds, stampLine

rawOsmPath = Path(__file__).resolve().parents[1] / "data" / "osm" / "raw.json"
cityJsonPath = Path(__file__).resolve().parents[1] / "frontend" / "public" / "city" / "islamabad.json"
simplifyTolerance = 1.5  # metres

roadClasses = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service"}
greenMap = {
    ("leisure", "park"): "park",
    ("leisure", "garden"): "garden",
    ("leisure", "pitch"): "pitch",
    ("landuse", "grass"): "grass",
    ("landuse", "meadow"): "grass",
    ("landuse", "village_green"): "grass",
    ("landuse", "recreation_ground"): "park",
    ("landuse", "forest"): "forest",
    ("natural", "wood"): "wood",
    ("natural", "scrub"): "scrub",
    ("natural", "grassland"): "grass",
}
waterways = {"river", "canal", "stream", "drain", "ditch"}


def roadClass(highway: str) -> str:
    base = highway.replace("_link", "")
    if base in roadClasses:
        return base
    if base in ("footway", "path", "pedestrian", "cycleway", "steps"):
        return "footway"
    return "other"


def greenKind(tags) -> str | None:
    for (key, value), kind in greenMap.items():
        if tags.get(key) == value:
            return kind
    return None


def number(value):
    try:
        return float(str(value).replace("m", "").strip())
    except ValueError:
        return None


def buildingHeight(wayId, tags) -> float:
    """height tag, else levels * 3.2, else a deterministic pseudo-random 6-18 m seeded by the way id."""
    height = number(tags.get("height", ""))
    if height is None and "building:levels" in tags:
        levels = number(tags["building:levels"])
        height = None if levels is None else levels * 3.2
    if height is None:
        height = random.Random(wayId).uniform(6, 18)
    return round(height, 1)


def projection(bounds=cityBounds):
    """Local metres: origin at the south-west corner, x = east, y = north (equirectangular)."""
    metresPerLat = 111320.0
    metresPerLon = 111320.0 * math.cos(math.radians((bounds["south"] + bounds["north"]) / 2))
    width = (bounds["east"] - bounds["west"]) * metresPerLon
    height = (bounds["north"] - bounds["south"]) * metresPerLat

    def toXY(lat, lon):
        return ((lon - bounds["west"]) * metresPerLon, (lat - bounds["south"]) * metresPerLat)

    return toXY, width, height


def clipSegment(p, q, width, height):
    """Liang-Barsky: the part of segment p-q inside [0,width] x [0,height], or None."""
    (x0, y0), (x1, y1) = p, q
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for direction, distance in ((-dx, x0), (dx, width - x0), (-dy, y0), (dy, height - y0)):
        if direction == 0:
            if distance < 0:
                return None
            continue
        t = distance / direction
        if direction < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def clipRuns(pts, width, height):
    """Segments clipped to the bbox and joined into runs; a run ends where the line leaves the bbox."""
    runs, current = [], []
    for p, q in zip(pts[:-1], pts[1:], strict=True):
        clipped = clipSegment(p, q, width, height)
        if clipped and current and current[-1] == clipped[0]:
            current.append(clipped[1])
            continue
        if len(current) >= 2:
            runs.append(current)
        current = list(clipped) if clipped else []
    if len(current) >= 2:
        runs.append(current)
    return runs


def loadFeatures(rawPath=rawOsmPath) -> dict:
    """Parses Overpass JSON into local-metre features (unsimplified)."""
    try:
        with open(rawPath, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"city data missing or invalid: {rawPath}") from err
    toXY, width, height = projection()
    features = {"roads": [], "buildings": [], "green": [], "water": [], "labels": [], "size": (width, height)}
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el.get("type") == "node":
            if "name" in tags and ("place" in tags or "amenity" in tags):
                x, y = toXY(el["lat"], el["lon"])
                kind = tags.get("place") or tags["amenity"]
                if 0 <= x <= width and 0 <= y <= height:
                    features["labels"].append({"name": tags["name"], "kind": kind, "x": x, "y": y})
            continue
        geometry = el.get("geometry")
        if el.get("type") != "way" or not geometry:
            continue
        pts = [toXY(p["lat"], p["lon"]) for p in geometry]
        closed = len(geometry) >= 4 and geometry[0] == geometry[-1]
        if "highway" in tags:
            cls, name = roadClass(tags["highway"]), tags.get("name", "")
            features["roads"] += [
                {"cls": cls, "name": name, "pts": run} for run in clipRuns(pts, width, height)
            ]
        elif "building" in tags and closed:
            features["buildings"].append({"h": buildingHeight(el["id"], tags), "pts": pts})
        elif greenKind(tags) and closed:
            features["green"].append({"kind": greenKind(tags), "pts": pts})
        elif "waterway" in tags and tags["waterway"] in waterways and not closed:
            features["water"].append({"kind": tags["waterway"], "pts": pts, "closed": False})
        elif closed and (tags.get("natural") == "water" or tags.get("waterway") == "riverbank"):
            features["water"].append({"kind": "water", "pts": pts, "closed": True})
    return features


def pointSegmentDistance(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def simplify(pts, tolerance=simplifyTolerance):
    """Douglas-Peucker, iterative. Keeps the endpoints (so closed rings stay closed)."""
    if len(pts) < 3:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        far, idx = 0.0, None
        for k in range(i + 1, j):
            distance = pointSegmentDistance(pts[k], pts[i], pts[j])
            if distance > far:
                far, idx = distance, k
        if idx is not None and far > tolerance:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, kept in zip(pts, keep, strict=True) if kept]


def rounded(pts):
    return [[round(x, 1), round(y, 1)] for x, y in pts]


def buildCityMap(rawPath=rawOsmPath, outPath=cityJsonPath) -> dict:
    """Writes the compact frontend city JSON (local metres, simplified) and returns the counts."""
    features = loadFeatures(rawPath)
    width, height = features["size"]
    roads = [
        {"cls": r["cls"], "name": r["name"], "pts": rounded(simplify(r["pts"]))} for r in features["roads"]
    ]
    roads = [r for r in roads if len(r["pts"]) >= 2]
    buildings = []
    for b in features["buildings"]:
        ring = simplify(b["pts"])
        if len(b["pts"]) >= 4 and len(ring) >= 4:
            buildings.append({"pts": rounded(ring), "h": b["h"]})
    green = [{"kind": g["kind"], "pts": rounded(simplify(g["pts"]))} for g in features["green"]]
    green = [g for g in green if len(g["pts"]) >= 4]
    water = [{"kind": w["kind"], "pts": rounded(simplify(w["pts"]))} for w in features["water"]]
    water = [w for w in water if len(w["pts"]) >= (4 if w.get("closed") else 2)]
    labels = [
        {"name": a["name"], "kind": a["kind"], "x": round(a["x"], 1), "y": round(a["y"], 1)}
        for a in features["labels"]
    ]
    doc = {
        "stamp": stampLine,
        "bounds": cityBounds,
        "size": {"width": round(width, 1), "height": round(height, 1)},
        "roads": roads,
        "buildings": buildings,
        "green": green,
        "water": water,
        "labels": labels,
    }
    outPath.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, separators=(",", ":"))
    outPath.write_text(text, encoding="utf-8")
    counts = {key: len(doc[key]) for key in ("roads", "buildings", "green", "water", "labels")}
    return {**counts, "bytes": len(text.encode("utf-8"))}
