import csv
import requests
import datetime
import logging
from typing import Optional, Dict, Set, Tuple, List
from data_sources.data_source import DataSource
from training.cleaning_pipeline import CleaningPipeline

logger = logging.getLogger(__name__)


class CommoditiesDataSource(DataSource):
    HOST = "https://reporting.trade-tariff.service.gov.uk"
    PATH = "/:service/reporting/:year/:month/:day/commodities_:service_:year_:month_:day.csv"
    DEFAULT_SERVICE = "uk"

    SID = 0
    COMMODITY_CODE = 1
    PRODUCT_LINE_SUFFIX = 2
    DESCRIPTION = 3
    START_DATE = 4
    END_DATE = 5
    INDENTATION = 6
    END_LINE = 7
    CLASS = 8
    ITEMIDPLUSPLS = 9
    HIERARCHY = 10

    TARGETED_LEVEL = "Commodity"

    def __init__(
        self,
        service: str = DEFAULT_SERVICE,
        cleaning_pipeline: Optional[CleaningPipeline] = None,
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

    def get_codes(self, digits: int) -> Dict[str, Set[str]]:
        commodities: Dict[str, Set[str]] = {}
        hierarchical_descriptions: Dict[str, str] = {}

        for index, row in enumerate(self._reader()):
            if index == 0:
                continue

            self._process_row(row, digits, commodities, hierarchical_descriptions)

        self._log_statistics(commodities)
        return commodities

    def _process_row(
        self,
        row: List[str],
        digits: int,
        commodities: Dict[str, Set[str]],
        hierarchical_descriptions: Dict[str, str],
    ) -> None:
        subheading: str = row[self.COMMODITY_CODE][:digits]
        description: str = row[self.DESCRIPTION]
        hierarchy: str = row[self.HIERARCHY]
        item_id_plus_pls: str = row[self.ITEMIDPLUSPLS]

        if row[self.CLASS] != self.TARGETED_LEVEL:
            hierarchical_descriptions[item_id_plus_pls] = description
        else:
            description = self._build_description(
                hierarchy, hierarchical_descriptions, description
            )
            result = self._filter_description(subheading, description)
            if result is not None:
                subheading, description = result
                self._add_to_commodities(subheading, description, commodities)

    def _build_description(
        self,
        hierarchy: str,
        hierarchical_descriptions: Dict[str, str],
        description: str,
    ) -> str:
        acc = ""
        for h_item_id_plus_plus in hierarchy.split(" "):
            if h_item_id_plus_plus in hierarchical_descriptions:
                acc += hierarchical_descriptions[h_item_id_plus_plus] + " "
        acc += description
        return acc

    def _filter_description(
        self,
        subheading: str,
        description: str,
    ) -> Optional[Tuple[str, str]]:
        if self._cleaning_pipeline:
            return self._cleaning_pipeline.filter(subheading, description)
        return subheading, description

    def _add_to_commodities(
        self,
        subheading: str,
        description: str,
        commodities: Dict[str, Set[str]],
    ) -> None:
        if subheading in commodities:
            commodities[subheading].add(description)
        else:
            commodities[subheading] = {description}

    def _log_statistics(self, commodities: Dict[str, Set[str]]) -> None:
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

    def _reader(self) -> csv.reader:
        response = requests.get(self._url)
        content = response.content.decode("utf-8").splitlines()
        return csv.reader(content)
