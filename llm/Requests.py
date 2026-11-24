# Dispatcher requests for this instance. Stops are 1..6 in core.Instance order.
# expected.tool None means the right answer is to refuse without calling a tool.
# checks: ["found", bool] | ["costAtMost", x] | ["withinBudget", bool] | ["orderPrefix", [stops]]
#         | ["terrainUnused", name] | ["savingAtLeast", x] | ["outcomeIn", [outcomes]]
requests = [
    {
        "id": 1,
        "text": "Find the cheapest route that delivers every parcel.",
        "expected": {"tool": "planRoute", "args": {}, "checks": [["found", True], ["costAtMost", 463]]},
    },
    {
        "id": 2,
        "text": "Use uniform cost search for the cheapest route.",
        "expected": {
            "tool": "planRoute",
            "args": {"algorithm": "ucs"},
            "checks": [["found", True], ["costAtMost", 463]],
        },
    },
    {
        "id": 3,
        "text": "Give me a fast greedy route.",
        "expected": {
            "tool": "planRoute",
            "args": {"algorithm": "greedy"},
            "checks": [["found", True], ["costAtMost", 513]],
        },
    },
    {
        "id": 4,
        "text": "Plan the route with BFS and tell me its cost.",
        "expected": {
            "tool": "planRoute",
            "args": {"algorithm": "bfs"},
            "checks": [["found", True], ["costAtMost", 525]],
        },
    },
    {
        "id": 5,
        "text": "Plan with A* using the plain Manhattan heuristic.",
        "expected": {
            "tool": "planRoute",
            "args": {"heuristic": "manhattanNearest"},
            "checks": [["found", True], ["costAtMost", 463]],
        },
    },
    {
        "id": 6,
        "text": "Start the route at cell [0, 0] and deliver everything.",
        "expected": {
            "tool": "planRoute",
            "args": {"start": [0, 0]},
            "checks": [["found", True], ["costAtMost", 453]],
        },
    },
    {
        "id": 7,
        "text": "Deliver stop 4 first, then the rest.",
        "expected": {
            "tool": "planRoute",
            "args": {"order": [4]},
            "checks": [["found", True], ["orderPrefix", [4]], ["costAtMost", 463]],
        },
    },
    {
        "id": 8,
        "text": "Plan a route that stays within a cost of 500.",
        "expected": {
            "tool": "planRoute",
            "args": {"maxCost": 500},
            "checks": [["found", True], ["withinBudget", True], ["costAtMost", 500]],
        },
    },
    {
        "id": 9,
        "text": "Plan a route that avoids mud.",
        "expected": {
            "tool": "planRoute",
            "args": {"avoidTerrain": ["mud"]},
            "checks": [["found", True], ["terrainUnused", "mud"], ["costAtMost", 797]],
        },
    },
    {
        "id": 10,
        "text": "Deliver stop 2 first, staying within 600.",
        "expected": {
            "tool": "planRoute",
            "args": {"order": [2], "maxCost": 600},
            "checks": [["found", True], ["orderPrefix", [2]], ["withinBudget", True]],
        },
    },
    {
        "id": 11,
        "text": "Deliver stop 3 first, then everything else.",
        "expected": {
            "tool": "planRoute",
            "args": {"order": [3]},
            "checks": [["found", True], ["orderPrefix", [3]], ["costAtMost", 598]],
        },
    },
    {
        "id": 12,
        "text": "Deliver stop 6 then stop 1.",
        "expected": {
            "tool": "planRoute",
            "args": {"order": [6, 1]},
            "checks": [["found", True], ["orderPrefix", [6, 1]], ["costAtMost", 696]],
        },
    },
    {
        "id": 13,
        "text": "Deliver stop 5 then stop 2.",
        "expected": {
            "tool": "planRoute",
            "args": {"order": [5, 2]},
            "checks": [["found", True], ["orderPrefix", [5, 2]], ["costAtMost", 710]],
        },
    },
    {
        "id": 14,
        "text": "How much do the teleport tunnels save between stop 1 and stop 4?",
        "expected": {
            "tool": "tunnelShortcut",
            "args": {"fromStop": 1, "toStop": 4},
            "checks": [["savingAtLeast", 24]],
        },
    },
    {
        "id": 15,
        "text": "Run one simulated episode with seed 7.",
        "expected": {
            "tool": "simulate",
            "args": {"seed": 7},
            "checks": [["outcomeIn", ["win", "draw", "loss"]]],
        },
    },
    {
        "id": 16,
        "text": "Plan a route that stays within a budget of 1.",
        "expected": {
            "tool": "planRoute",
            "args": {"maxCost": 1},
            "checks": [["found", True], ["withinBudget", False]],
        },
    },
    {
        "id": 17,
        "text": "Deliver stop 3 first and avoid mud.",
        "expected": {
            "tool": "planRoute",
            "args": {"order": [3], "avoidTerrain": ["mud"]},
            "checks": [
                ["found", True],
                ["orderPrefix", [3]],
                ["terrainUnused", "mud"],
                ["costAtMost", 1007],
            ],
        },
    },
    {
        "id": 18,
        "text": "Deliver stop 4 first, avoid mud, and stay within 900.",
        "expected": {
            "tool": "planRoute",
            "args": {"order": [4], "avoidTerrain": ["mud"], "maxCost": 900},
            "checks": [
                ["found", True],
                ["orderPrefix", [4]],
                ["terrainUnused", "mud"],
                ["withinBudget", True],
            ],
        },
    },
    {
        "id": 19,
        "text": "Plan a route that avoids ford.",
        "expected": {
            "tool": "planRoute",
            "args": {"avoidTerrain": ["ford"]},
            "checks": [["found", False]],
        },
    },
    {
        "id": 20,
        "text": "Deliver stop 7 first.",
        "expected": {"tool": None, "args": {}, "checks": []},
    },
]
