from llm.Tools import toolSpecs


def specFor(name):
    return next((spec for spec in toolSpecs if spec["name"] == name), None)


def checkValue(key, value, schema):
    """Type and enum checks for one argument. Returns (coercedValue, errors)."""
    kind = schema["type"]
    if kind == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)
    if kind == "string" and isinstance(value, str) and "enum" in schema and value.lower() in schema["enum"]:
        value = value.lower()
    if kind == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif kind == "string":
        ok = isinstance(value, str)
    else:
        ok = isinstance(value, list)
    if not ok:
        return value, [f"argument {key!r} must be {kind}, got {value!r}"]
    if "enum" in schema and value not in schema["enum"]:
        return value, [f"argument {key!r} must be one of {schema['enum']}, got {value!r}"]
    if kind != "array":
        return value, []
    errors = []
    if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 10**9):
        errors.append(f"argument {key!r} has the wrong number of items: {value!r}")
    if schema.get("uniqueItems") and len(set(map(str, value))) != len(value):
        errors.append(f"argument {key!r} must not repeat items")
    coerced = []
    for item in value:
        fixedItem, itemErrors = checkValue(key, item, schema["items"])
        coerced.append(fixedItem)
        errors += itemErrors
    return coerced, errors


def semanticErrors(key, value, instance) -> list[str]:
    count = len(instance.stops)
    if key in ("order", "fromStop", "toStop"):
        numbers = value if isinstance(value, list) else [value]
        bad = [n for n in numbers if not 1 <= n <= count]
        if bad:
            return [f"stop {bad[0]} does not exist; stops are numbered 1..{count}"]
    if key == "start":
        rows, cols = instance.grid.shape
        r, c = value
        if not (0 <= r < rows and 0 <= c < cols):
            return [f"start {value} is outside the {rows}x{cols} grid"]
        if instance.grid[r, c] < 0:
            return [f"start {value} is a blocked cell"]
    if key == "maxCost" and value <= 0:
        return ["maxCost must be positive"]
    if key == "steps" and value < 1:
        return ["steps must be at least 1"]
    return []


def validateCall(name, args, instance):
    """Returns (ok, errors, fixedArgs). Errors are readable strings for the LLM's Observation."""
    spec = specFor(name)
    if spec is None:
        known = [s["name"] for s in toolSpecs]
        return False, [f"unknown tool {name!r}; available tools are {known}"], {}
    if not isinstance(args, dict):
        return False, ["args must be a JSON object"], {}
    schema = spec["parameters"]
    props = schema["properties"]
    errors = [
        f"invented argument {key!r}; allowed arguments are {sorted(props)}"
        for key in args
        if key not in props
    ]
    errors += [f"missing required argument {key!r}" for key in schema["required"] if key not in args]
    fixed = {}
    for key, value in args.items():
        if key not in props:
            continue
        coerced, problems = checkValue(key, value, props[key])
        fixed[key] = coerced
        errors += problems
        if not problems:
            errors += semanticErrors(key, coerced, instance)
    return not errors, errors, fixed
