import numpy as np

tunnelCost = 2.0  # cost of the teleport action


def makeTunnels(instance, pairs=3) -> dict:
    """Deterministic symmetric teleport map a->b, b->a over free cells that are not the start or a stop."""
    rng = np.random.default_rng(int(instance.instanceId, 16))
    excluded = {instance.start, *instance.stops}
    free = [tuple(map(int, cell)) for cell in np.argwhere(instance.grid >= 0)]
    free = [cell for cell in free if cell not in excluded]
    picks = rng.choice(len(free), 2 * pairs, replace=False)
    cells = [free[i] for i in picks]
    tunnels = {}
    for a, b in zip(cells[::2], cells[1::2], strict=True):
        tunnels[a] = b
        tunnels[b] = a
    return tunnels
