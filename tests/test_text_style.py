from __future__ import annotations

import unittest
import subprocess
from pathlib import Path


class TextStyleTests(unittest.TestCase):
    def test_tracked_text_has_no_prohibited_dash(self) -> None:
        root = Path(__file__).resolve().parents[1]
        extensions = {".md", ".py", ".yaml", ".yml", ".tsv", ".txt", ".cff"}
        offenders = []
        excluded = {".git", "work", "data", "outputs"}
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
        for relative_name in filter(None, tracked):
            path = root / relative_name
            relative = path.relative_to(root)
            if any(part in excluded for part in relative.parts):
                continue
            if path.is_file() and path.suffix.lower() in extensions:
                text = path.read_text(encoding="utf-8")
                if "\N{EM DASH}" in text or "\N{EN DASH}" in text:
                    offenders.append(str(relative))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
