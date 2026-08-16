import tempfile
import unittest
from pathlib import Path

from indexer.config import load_data_sources


class ConfigLoaderTests(unittest.TestCase):
    def test_load_data_sources_reads_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "data_sources.yaml"
            config_path.write_text(
                "adventurelookup:\n"
                "  base_url: \"https://example.com/api\"\n"
                "  seed: 99\n",
                encoding="utf-8",
            )

            config = load_data_sources(config_path)

            self.assertEqual(config["adventurelookup"]["base_url"], "https://example.com/api")
            self.assertEqual(config["adventurelookup"]["seed"], 99)

    def test_load_data_sources_uses_default_project_config(self):
        config = load_data_sources()
        self.assertIn("adventurelookup", config)
        self.assertIn("base_url", config["adventurelookup"])
        self.assertIn("seed", config["adventurelookup"])


if __name__ == "__main__":
    unittest.main()
