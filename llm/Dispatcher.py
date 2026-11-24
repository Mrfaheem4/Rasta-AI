import json

from llm.Tools import baseInstance, tools, toolSpecs
from llm.Validation import validateCall


def systemPrompt() -> str:
    count = len(baseInstance().stops)
    return (
        f"You dispatch a delivery robot. Stops are numbered 1..{count} (humans say 'stop 3'). "
        "Reply with one step per turn, in exactly one of these forms:\n"
        'Thought: <one short sentence>\nAction: {"tool": "<name>", "args": {...}}\n'
        "or, when you have the answer:\nThought: <one short sentence>\nFinal: <answer>\n"
        "Use only the arguments listed in the schema. Available tools (JSON schema):\n"
        + json.dumps(toolSpecs, indent=1)
    )


def parseReply(reply: str):
    """Returns (thought, action dict or None, final text or None, error or None)."""
    if "Action:" in reply:
        head, rest = reply.split("Action:", 1)
        thought = head.replace("Thought:", "").strip()
        brace = rest.find("{")
        if brace < 0:
            return thought, None, None, "Action has no JSON object"
        try:
            action, _ = json.JSONDecoder().raw_decode(rest[brace:])
        except json.JSONDecodeError as err:
            return thought, None, None, f"invalid JSON in Action: {err}"
        if not isinstance(action, dict) or "tool" not in action:
            return thought, None, None, 'Action must be {"tool": ..., "args": {...}}'
        return thought, action, None, None
    if "Final:" in reply:
        head, final = reply.split("Final:", 1)
        return head.replace("Thought:", "").strip(), None, final.strip(), None
    return "", None, None, "reply has neither Action nor Final"


def emptyCounters() -> dict:
    return {"toolCalls": 0, "wrongTool": 0, "inventedArgs": 0, "invalidArgs": 0, "parseErrors": 0}


def runDispatcher(request: dict, generate, validate: bool = True, maxTurns: int = 6) -> dict:
    messages = [
        {"role": "system", "content": systemPrompt()},
        {"role": "user", "content": request["text"]},
    ]
    counters = emptyCounters()
    trace, results, final = [], [], None
    for _ in range(maxTurns):
        reply = generate(messages)
        messages.append({"role": "assistant", "content": reply})
        thought, action, answer, error = parseReply(reply)
        if error:
            counters["parseErrors"] += 1
            observation = f"parse error: {error}. Reply with Thought, then Action JSON, or Final."
        elif action is None:
            final = answer
            trace.append({"thought": thought, "action": None, "observation": None})
            break
        elif action["tool"] not in tools:
            counters["wrongTool"] += 1
            observation = f"unknown tool {action['tool']!r}; available: {sorted(tools)}"
        else:
            name, args = action["tool"], action.get("args", {})
            ok, errors, fixed = validateCall(name, args, baseInstance()) if validate else (True, [], args)
            if not ok:
                if any("invented" in e for e in errors):
                    counters["inventedArgs"] += 1
                if any("invented" not in e for e in errors):
                    counters["invalidArgs"] += 1
                observation = "validation failed, the tool was not run: " + "; ".join(errors)
            else:
                try:
                    result = tools[name](**fixed)
                except TypeError as err:
                    counters["inventedArgs" if "unexpected keyword" in str(err) else "invalidArgs"] += 1
                    observation = f"bad arguments: {err}"
                else:
                    counters["toolCalls"] += 1
                    results.append({"tool": name, "args": fixed, "result": result})
                    observation = json.dumps(result)
                    trace.append({"thought": thought, "action": action, "observation": result})
                    messages.append({"role": "user", "content": "Observation: " + observation})
                    continue
        trace.append({"thought": thought, "action": action, "observation": observation})
        messages.append({"role": "user", "content": "Observation: " + observation})
    return {"final": final, "trace": trace, "results": results, **counters}


def runDirect(request: dict) -> dict:
    """Baseline: call the planner directly with the expected arguments (no LLM)."""
    expected = request["expected"]
    run = {"final": None, "trace": [], "results": [], **emptyCounters()}
    if expected["tool"] is None:
        return run
    result = tools[expected["tool"]](**expected["args"])
    run["toolCalls"] = 1
    run["results"].append({"tool": expected["tool"], "args": expected["args"], "result": result})
    run["trace"].append({"thought": "direct", "action": expected, "observation": result})
    return run


def checkPasses(check, result: dict) -> bool:
    kind, value = check[0], (check[1] if len(check) > 1 else None)
    if kind == "found":
        return result.get("found") == value
    if kind == "costAtMost":
        return result.get("cost") is not None and result["cost"] <= value
    if kind == "withinBudget":
        return result.get("withinBudget") == value
    if kind == "orderPrefix":
        return result.get("stopVisitOrder", [])[: len(value)] == value
    if kind == "terrainUnused":
        return result.get("terrainSteps", {}).get(value, 0) == 0
    if kind == "savingAtLeast":
        return "saving" in result and result["saving"] >= value
    if kind == "outcomeIn":
        return result.get("outcome") in value
    return False


def scoreRun(request: dict, run: dict) -> dict:
    expected = request["expected"]
    violations = []
    if expected["tool"] is None:
        if run["results"]:
            violations.append("called a tool although the request should be refused")
        return {
            "success": not violations,
            "cost": None,
            "toolCalls": run["toolCalls"],
            "violations": violations,
        }
    matching = [r for r in run["results"] if r["tool"] == expected["tool"]]
    if not matching:
        violations.append(f"no successful call to {expected['tool']}")
        return {"success": False, "cost": None, "toolCalls": run["toolCalls"], "violations": violations}
    result = matching[-1]["result"]
    for check in expected["checks"]:
        if not checkPasses(check, result):
            violations.append(f"failed {check}")
    return {
        "success": not violations,
        "cost": result.get("cost"),
        "toolCalls": run["toolCalls"],
        "violations": violations,
    }
