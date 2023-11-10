from pathlib import Path
from data_sources.data_source import DataSource
from data_sources.trade_tariff import TradeTariffDataSource
from data_sources.tradesets import BasicCSVDataSource

class TrainingDataLoader:
    def __init__(self, source_dir: Path|None = None) -> None:
        if source_dir is None:
            cwd = Path(__file__).resolve().parent
            self._source_dir = cwd / "raw_source_data"
        else:
            self._source_dir = source_dir

    def fetch_data(self, data_sources: list[DataSource], digits: int = 8):
        # This will store a map of the original subheading to the category number
        subheadings_map = {}

        subheadings = []
        texts = []
        labels = []

        for data_source in data_sources:
            print(f"Retrieving data from source: {data_source.get_description()}")

            data = data_source.get_codes(digits)

            for subheading, descriptions in data.items():
                if subheading in subheadings_map:
                    label = subheadings_map[subheading]
                else:
                    label = len(subheadings)
                    subheadings_map[subheading] = label
                    subheadings.append(subheading)

                for description in descriptions:
                    texts.append(description)
                    labels.append(label)

        return (
            texts,
            labels,
            subheadings,
        )
