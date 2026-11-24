# Conventions

- **Names:** PascalCase for classes, protocols and file names (`AStar.py`, `class SearchProblem`). camelCase for variables, functions, methods and objects (`costMap`, `def planRoute`). Folders are lowercase.
- **Less code:** prefer NumPy and the standard library, avoid wrapper classes that add nothing, and skip comments that just restate the code.
- **Algorithms:** every search and game algorithm lives in `algo/` and works on the general `SearchProblem` / `GameProblem` interfaces. Nothing outside `algo/` reimplements one.
- **Stamp:** every printed table, log, plot and UI screen shows `core.Config.stampLine`.
- **Formatting:** `ruff format` and `ruff check` must pass.
- **Git:** one small commit per change; branch names like `feat/astar`; one PR per feature.
