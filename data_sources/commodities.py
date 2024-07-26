import csv
import requests
import datetime
import logging

from data_sources.data_source import DataSource
from training.cleaning_pipeline import CleaningPipeline

logger = logging.getLogger(__name__)


class CommoditiesDataSource(DataSource):
    HOST = "https://reporting.trade-tariff.service.gov.uk"
    PATH = "/:service/reporting/:year/:month/:day/commodities_:service_:year_:month_:day.csv"
    DEFAULT_SERVICE = "uk"

    def __init__(
        self,
        service: str = DEFAULT_SERVICE,
        cleaning_pipeline: CleaningPipeline | None = None,
        authoritative: bool = False,
        creates_codes: bool = False,
        multiplier: int = 1,
    ):
        url = f"{self.HOST}{self.PATH}"
        today = datetime.datetime.now()
        year = str(today.year)
        month = str(today.month).rjust(2, "0")
        day = str(today.day).rjust(2, "0")

        url = (
            url.replace(":service", service)
            .replace(":year", year)
            .replace(":month", month)
            .replace(":day", day)
        )

        super().__init__(
            description=f"Commodities data source from {str(url)}",
            authoritative=authoritative,
            creates_codes=creates_codes,
            multiplier=multiplier,
        )

        self._url = url
        self._cleaning_pipeline = cleaning_pipeline

    def get_codes(self, digits: int) -> dict[str, set[str]]:
        commodities: dict[str, set[str]] = {}

        for index, row in enumerate(self._reader()):
            if index == 0:
                continue

            subheading = row[1][:digits]
            description = row[3]

            result = None

            if self._cleaning_pipeline:
                result = self._cleaning_pipeline.filter(subheading, description)

                if result is None:
                    continue

            if result is not None:
                subheading, description = result

            if subheading in commodities:
                commodities[subheading].add(description)
            else:
                commodities[subheading] = {description}

        total_descriptions = sum(
            len(descriptions) for descriptions in commodities.values()
        )
        unique_descriptions = len(
            {
                description
                for descriptions in commodities.values()
                for description in descriptions
            }
        )
        logger.info(
            f"Loaded {len(commodities)} subheadings with {unique_descriptions} unique descs and {total_descriptions} total descs"
        )
        return commodities

    def _reader(self):
        response = requests.get(self._url)
        content = response.content.decode("utf-8").splitlines()

        return csv.reader(content)
