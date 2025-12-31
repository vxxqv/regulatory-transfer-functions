"""Strict interval liftover with explicit ambiguity accounting."""

from __future__ import annotations

from pyliftover import LiftOver


def lift_endpoint(
    converter: LiftOver, chromosome: str, position: int
) -> tuple[tuple[str, int, str, float] | None, int, str]:
    source = str(chromosome)
    source = source if source.startswith("chr") else f"chr{source}"
    hits = converter.convert_coordinate(source, int(position))
    if not hits:
        return None, 0, "unmapped_endpoint"
    best_score = max(float(hit[3]) for hit in hits)
    best = [hit for hit in hits if float(hit[3]) == best_score]
    unique = {(str(hit[0]), int(hit[1]), str(hit[2])) for hit in best}
    if len(unique) != 1:
        return None, len(hits), "ambiguous_best_endpoint"
    target_chromosome, target_position, strand = next(iter(unique))
    return (
        (target_chromosome.removeprefix("chr"), target_position, strand, best_score),
        len(hits),
        "mapped",
    )


def lift_interval(
    converter: LiftOver, chromosome: str, start: int, end: int
) -> dict[str, object]:
    if int(end) <= int(start):
        return {
            "status": "invalid_source_interval",
            "first_candidates": 0,
            "last_candidates": 0,
        }
    first, first_candidates, first_status = lift_endpoint(converter, chromosome, int(start))
    last, last_candidates, last_status = lift_endpoint(converter, chromosome, int(end) - 1)
    base = {
        "first_candidates": first_candidates,
        "last_candidates": last_candidates,
    }
    if first is None or last is None:
        return {**base, "status": first_status if first is None else last_status}
    if first[0] != last[0]:
        return {**base, "status": "endpoint_chromosome_mismatch"}
    if first[2] != last[2]:
        return {**base, "status": "endpoint_orientation_mismatch"}
    mapped = sorted([first[1], last[1]])
    return {
        **base,
        "status": "mapped",
        "chromosome": first[0],
        "start": mapped[0],
        "end": mapped[1] + 1,
        "strand": first[2],
        "first_chain_score": first[3],
        "last_chain_score": last[3],
    }
