import pandas as pd

from src.ingest.prepare_encode_crispri import as_boolean


def test_encode_boolean_parser_is_strict() -> None:
    values = pd.Series([True, False, "TRUE", "false", None, "unknown"])
    assert as_boolean(values).tolist() == [True, False, True, False, False, False]
