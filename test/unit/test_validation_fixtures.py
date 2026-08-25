import unittest

from khepri.validation import case_registry, fixture_path, load_manifest


class FixtureTests(unittest.TestCase):
    def test_literature_manifests_have_provenance_and_honest_data_scope(self):
        luder = load_manifest("luder_2020/manifest.json")
        lou = load_manifest("lou_2021/manifest.json")
        fan = load_manifest("suh_fan_2003/manifest.json")
        self.assertIn("doi", luder["source"])
        self.assertIn("uncertainty", luder)
        self.assertIsNone(luder["data_file"])
        self.assertEqual(lou["observable"], "total far-field transmitted power for incident RCP")
        self.assertEqual(fan["geometry"]["hole_radius_over_a"], 0.4)
        self.assertIsNone(fan["data_file"])

    def test_every_registered_case_cites_a_doi_or_arxiv_source(self):
        for case in case_registry().values():
            self.assertTrue(case.references, case.slug)
            for reference in case.references:
                self.assertTrue(
                    "doi.org/" in reference.url or "arxiv.org/abs/" in reference.url,
                    f"{case.slug}: {reference.url}",
                )

    def test_fixture_paths_cannot_escape_reference_directory(self):
        with self.assertRaises(ValueError):
            fixture_path("../models.py")


if __name__ == "__main__":
    unittest.main()
