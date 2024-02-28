import unittest
from pathlib import Path
from training.prepare_data import TrainingDataLoader


class Test_training_data_loader(unittest.TestCase):
    def setUp(self):
        self.mock_source_dir = Path("/mock/source/dir")

    def test_init_default_source_dir(self):
        loader = TrainingDataLoader()
        expected_dir = Path(__file__).resolve().parent / "raw_source_data"
        self.assertEqual(loader._source_dir, expected_dir)

    def test_init_custom_source_dir(self):
        loader = TrainingDataLoader(source_dir=self.mock_source_dir)
        self.assertEqual(loader._source_dir, self.mock_source_dir)

    @unittest.mock.patch("data_sources.data_source")
    def test_fetch_data(self, MockDataSource):
        """Test fetch_data correctly processes data from data sources."""
        # Setup mock DataSource behavior
        mock_data_source = MockDataSource()
        mock_data_source.get_description.return_value = "Mock DataSource"
        mock_data_source.get_codes.return_value = {
            "12345678": ["Description 1", "Description 2"]
        }

        loader = TrainingDataLoader(source_dir=self.mock_source_dir)
        texts, labels, subheadings = loader.fetch_data([mock_data_source], digits=8)

        # Verify the returned data
        self.assertEqual(texts, ["Description 1", "Description 2"])
        self.assertEqual(labels, [0, 0])
        self.assertEqual(subheadings, ["12345678"])
        # Verify DataSource methods were called as expected
        mock_data_source.get_description.assert_called_once()
        mock_data_source.get_codes.assert_called_once_with(8)


if __name__ == "__main__":
    unittest.main()
