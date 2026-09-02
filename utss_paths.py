"""utss_paths.py - anchor every entry point to the repository root.

Every generator in this project reads and writes with paths relative to the
repository root ("04_solution/...", "06_validation/plots/...") and imports the
solver with sys.path.insert(0, "solver").  Both are relative to the CURRENT
WORKING DIRECTORY, not to the script, so running

    cd solver && python3 ../gen_validation.py

did not fail - it silently created a second, empty 06_validation/plots inside
solver/ and wrote nothing where the report reads from.  One such directory was
found in the tree.

Importing this module first makes the location of the script, not the shell's
working directory, decide where the project lives:

    import utss_paths            # noqa: F401  (side effect is the point)

It puts the repository root and solver/ on sys.path and changes the working
directory to the root, so every existing relative path in every generator keeps
working unchanged, from any directory.

ROOT and p() are exported for code that would rather build an absolute path
than rely on the working directory.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(ROOT, "solver")

for _d in (SOLVER, ROOT):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# The generators are written against the root; hold them to it.
if os.path.abspath(os.getcwd()) != ROOT:
    os.chdir(ROOT)


def p(*parts):
    """Absolute path inside the repository."""
    return os.path.join(ROOT, *parts)
