import json
from pathlib import Path
import tempfile
import unittest

from khepri.validation import (
    SequentialRunner,
    ThinFilmCase,
    write_json_run,
    write_markdown_run,
)


class ReportingTests(unittest.TestCase):
    def test_json_is_primary_and_markdown_links_to_it(self):
        case = ThinFilmCase()
        run = SequentialRunner().run(case)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = write_json_run(case, run, root / "thin-film.json")
            report = write_markdown_run(
                case, run, root / "thin-film.md", raw_json=raw
            )
            document = json.loads(raw.read_text(encoding="utf-8"))
            markdown = report.read_text(encoding="utf-8")
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["case_slug"], "thin-film")
        self.assertIn("thin-film.json", markdown)
        self.assertIn("Configuration results", markdown)


if __name__ == "__main__":
    unittest.main()
