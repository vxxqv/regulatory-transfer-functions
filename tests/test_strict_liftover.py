from __future__ import annotations

from analyses.common.strict_liftover import lift_interval


class FakeLiftOver:
    def __init__(self, mapping: dict[tuple[str, int], list[tuple[str, int, str, float]]]):
        self.mapping = mapping

    def convert_coordinate(self, chromosome: str, position: int):
        return self.mapping.get((chromosome, position), [])


def test_strict_liftover_maps_half_open_interval() -> None:
    converter = FakeLiftOver(
        {
            ("chr1", 10): [("chr1", 110, "+", 8.0)],
            ("chr1", 19): [("chr1", 119, "+", 8.0)],
        }
    )
    result = lift_interval(converter, "1", 10, 20)
    assert result["status"] == "mapped"
    assert (result["chromosome"], result["start"], result["end"]) == ("1", 110, 120)
    assert result["first_candidates"] == result["last_candidates"] == 1


def test_strict_liftover_rejects_tied_best_endpoints() -> None:
    converter = FakeLiftOver(
        {
            ("chr1", 10): [("chr1", 110, "+", 8.0), ("chr2", 210, "+", 8.0)],
            ("chr1", 19): [("chr1", 119, "+", 8.0)],
        }
    )
    result = lift_interval(converter, "1", 10, 20)
    assert result["status"] == "ambiguous_best_endpoint"
    assert result["first_candidates"] == 2


def test_strict_liftover_rejects_orientation_and_chromosome_mismatch() -> None:
    orientation = FakeLiftOver(
        {
            ("chr1", 10): [("chr1", 110, "+", 8.0)],
            ("chr1", 19): [("chr1", 119, "-", 8.0)],
        }
    )
    chromosome = FakeLiftOver(
        {
            ("chr1", 10): [("chr1", 110, "+", 8.0)],
            ("chr1", 19): [("chr2", 119, "+", 8.0)],
        }
    )
    assert lift_interval(orientation, "1", 10, 20)["status"] == "endpoint_orientation_mismatch"
    assert lift_interval(chromosome, "1", 10, 20)["status"] == "endpoint_chromosome_mismatch"
