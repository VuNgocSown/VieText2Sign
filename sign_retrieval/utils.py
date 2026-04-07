"""Utility functions for sign retrieval"""
from typing import List, Tuple


def find_non_overlapping_matches(matches: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """Select non-overlapping matches, prioritizing longer matches"""
    if not matches:
        return []
    
    sorted_matches = sorted(matches, key=lambda x: (-(x[2] - x[1]), x[1]))
    
    selected = []
    covered = set()
    
    for match in sorted_matches:
        gloss, start, end = match
        positions = set(range(start, end))
        if not positions.intersection(covered):
            selected.append(match)
            covered.update(positions)
    
    selected.sort(key=lambda x: x[1])
    return selected
