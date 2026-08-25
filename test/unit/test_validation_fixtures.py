import unittest

from khepri.validation import fixture_path, load_manifest


class FixtureTests(unittest.TestCase):
    def test_literature_manifests_have_provenance_and_honest_data_scope(self):
        luder = load_manifest("luder_2020/manifest.json")
        lou = load_manifest("lou_2021/manifest.json")
        self.assertIn("doi", luder["source"])
        self.assertIn("uncertainty", luder)
        self.assertIsNone(luder["data_file"])
        self.assertEqual(lou["observable"], "total far-field transmitted power for incident RCP")

    def test_fixture_paths_cannot_escape_reference_directory(self):
        with self.assertRaises(ValueError):
            fixture_path("../models.py")


if __name__ == "__main__":
    unittest.main()
