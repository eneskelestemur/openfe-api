"""Connectivity of a ligand network, independent of how its edges were obtained."""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["connected_components", "unreachable_names"]


def connected_components(names: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[set[str]]:
    """Group ligand names into the components its edges connect, smallest first.

    Args:
        names: Every ligand in the network, including any with no edges.
        edges: Ligand name pairs.

    Returns:
        One set per component, sorted by size.
    """
    neighbors: dict[str, set[str]] = {name: set() for name in names}
    for first, second in edges:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)

    seen: set[str] = set()
    groups: list[set[str]] = []
    for start in neighbors:
        if start in seen:
            continue
        group: set[str] = set()
        queue = [start]
        while queue:
            current = queue.pop()
            if current in group:
                continue
            group.add(current)
            queue.extend(neighbors[current] - group)
        seen |= group
        groups.append(group)

    groups.sort(key=len)
    return groups


def unreachable_names(names: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[str]:
    """Return the ligands outside the largest connected component.

    These get no comparable free energy, because nothing ties them to the rest of the
    network.

    Args:
        names: Every ligand in the network.
        edges: Ligand name pairs.

    Returns:
        The orphaned names, sorted.
    """
    groups = connected_components(names, edges)
    if len(groups) < 2:
        return []
    return sorted(name for group in groups[:-1] for name in group)
