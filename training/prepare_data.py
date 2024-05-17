import logging
import typing
from data_sources.data_source import DataSource


class TrainingData(typing.NamedTuple):
    text_values: list[str]
    subheadings: list[str]
    texts: list[int]
    labels: list[int]


class TrainingDataLoader:
    def __init__(
        self,
        logger: logging.Logger = logging.getLogger("training_data_loader"),
    ) -> None:
        self._logger = logger

    def fetch_data(
        self,
        data_sources: list[DataSource],
        digits: int = 8,
    ) -> TrainingData:
        unique_texts: list[str] = []
        unique_text_map: typing.Dict[str, int] = {}

        subheadings: list[str] = []
        subheadings_map: typing.Dict[str, int] = {}

        labels = list[int]()
        texts = list[int]()

        authoritative_texts = {}

        data_sources_with_data = [
            (data_source, data_source.get_codes(digits)) for data_source in data_sources
        ]

        # Go through all the code creating data sources and add them to the subheadings and the map
        for data_source, data in data_sources_with_data:
            if data_source.creates_codes:
                self._logger.info(
                    f"📇  Getting codes from code creating data source: {data_source.description}"
                )
                for subheading, descriptions in data.items():
                    if subheading not in subheadings_map:
                        subheading_idx = len(subheadings)
                        subheadings_map[subheading] = subheading_idx
                        subheadings.append(subheading)

        print(f"Found {len(subheadings)} subheadings")

        # Go through all the authoritative data sources and store the descriptions against the codes
        for data_source, data in data_sources_with_data:
            if data_source.authoritative:
                self._logger.info(
                    f"📇  Getting authoritative description to code mappings from authoritative data source: {data_source.description}"
                )
                for subheading, descriptions in data.items():
                    if subheading in subheadings_map:
                        for description in descriptions:
                            if description not in authoritative_texts:
                                authoritative_texts[description] = subheading
                            else:
                                if authoritative_texts[description] != subheading:
                                    self._logger.warn(
                                        f"❗ Ambiguous codes for '{description}' from multiple authoritative data sources."
                                    )
                                    self._logger.warn(
                                        f"❗ Previous code was {authoritative_texts[description]}."
                                    )
                                    self._logger.warn(f"❗ This code is {subheading}.")
                                    self._logger.warn(
                                        f"❗ Current data source is {data_source.description}."
                                    )

        # Secondary data sources do not extend the commodity code list. If an unknown commodity code is encountered here then we ignore it
        invalid_subheading_count = 0
        incorrect_code_for_description_count = 0

        for data_source, data in data_sources_with_data:
            self._logger.info(
                f"🗄️  Processing data from data source: {data_source.description}"
            )

            for subheading, descriptions in data.items():
                if subheading in subheadings_map:
                    subheading_idx = subheadings_map[subheading]
                else:
                    self._logger.debug(f"Subheading {subheading} not found - skipping")
                    invalid_subheading_count += 1
                    continue

                for description in descriptions:
                    this_subheading_idx = subheading_idx

                    # If the description already has an authoritative subheading then we'll use that instead
                    if (
                        description in authoritative_texts
                        and authoritative_texts[description] != subheading
                    ):
                        this_subheading_idx = subheadings_map[
                            authoritative_texts[description]
                        ]
                        incorrect_code_for_description_count += 1

                    if description in unique_text_map:
                        unique_text_idx = unique_text_map[description]
                    else:
                        unique_text_idx = len(unique_texts)
                        unique_text_map[description] = unique_text_idx
                        unique_texts.append(description)

                    labels.extend([this_subheading_idx] * data_source.multiplier)
                    texts.extend([unique_text_idx] * data_source.multiplier)

        self._logger.info(
            f"ℹ️  {invalid_subheading_count} entries with invalid subheadings were skipped"
        )
        self._logger.info(
            f"ℹ️  {incorrect_code_for_description_count} descriptions were overridden with an authoritative one"
        )

        return TrainingData(unique_texts, subheadings, texts, labels)
