import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from os import PathLike
from typing import Any, Dict, List, Optional, Set, Union

import dill

from data_sources.data_source import DataSource
from training.cleaning_pipeline import CleaningPipeline

logger = logging.getLogger(__name__)


def generate_chunk_wrapper(serialized_args: bytes) -> Dict[str, Set[str]]:
    """Wrapper function to deserialize arguments and do any processing."""
    # Deserialize the arguments
    cleaning_pipeline_data, chunk, code_col, description_col, digits = dill.loads(
        serialized_args
    )

    # Reconstruct the cleaning pipeline
    if cleaning_pipeline_data:
        cleaning_pipeline = CleaningPipeline.from_serialized_data(
            cleaning_pipeline_data
        )
    else:
        cleaning_pipeline = None

    # Process the chunk
    codes: dict[str, set[str]] = {}

    for row in chunk:
        subheading = row[code_col].replace(" ", "")[:digits]
        description = row[description_col].strip().lower()

        if cleaning_pipeline:
            subheading, description, _meta = cleaning_pipeline.filter(
                subheading, description
            )

            if subheading is None or description is None:
                continue

        if subheading in codes:
            codes[subheading].add(description)
        else:
            codes[subheading] = {description}

    return codes


class BasicCSVDataSource(DataSource):
    def __init__(
        self,
        filename: Union[str, PathLike],
        cleaning_pipeline: Optional[CleaningPipeline] = None,
        code_col: int = 0,
        description_col: int = 1,
        encoding: str = "utf-8",
        authoritative: bool = False,
        creates_codes: bool = False,
        multiplier: int = 1,
    ) -> None:
        super().__init__(
            description=f"CSV data source from {str(filename)}",
            authoritative=authoritative,
            creates_codes=creates_codes,
            multiplier=multiplier,
            cleaning_pipeline=cleaning_pipeline,
        )
        self.filename = filename
        self._code_col = code_col
        self._description_col = description_col
        self._encoding = encoding

    def get_codes(self, digits: int) -> dict[str, set[str]]:
        with open(self.filename, mode="r", encoding=self._encoding) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # skip the first line (header)
            code_data = list(csv_reader)

        chunks = self._chunk_rows(code_data, self._max_workers())
        all_results = self._do_work(chunks, digits)
        codes = self._merge_results(all_results)
        total_descriptions = sum(len(descriptions) for descriptions in codes.values())
        unique_descriptions = len(
            {
                description
                for descriptions in codes.values()
                for description in descriptions
            }
        )
        logger.info(
            f"Loaded {len(codes)} unique subheadings with {unique_descriptions} unique descriptions and {total_descriptions} total descriptions from {os.path.relpath(self.filename)}"
        )

        return codes

    def get_codes_for_cleaning_report(self, digits: int) -> List[Any]:
        with open(self.filename, mode="r", encoding=self._encoding) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            code_data = list(csv_reader)

        codes = []
        for row in code_data:
            subheading = row[self._code_col].replace(" ", "")[:digits]
            description = row[self._description_col].strip().lower()
            uncleaned_subheading = row[self._code_col]
            uncleaned_description = row[self._description_col]

            if self.cleaning_pipeline:
                subheading, description, meta = self.cleaning_pipeline.filter(
                    subheading, description
                )
                result = [
                    subheading,
                    description,
                    uncleaned_subheading,
                    uncleaned_description,
                    meta,
                ]
                codes.append(result)

        return codes

    def _do_work(
        self, chunks: List[List[List[str]]], digits: int
    ) -> List[dict[str, set[str]]]:
        all_results = []
        with ProcessPoolExecutor(self._max_workers()) as executor:
            # Serialize only the necessary components using dill
            serialized_tasks = [
                dill.dumps(
                    (
                        (
                            self.cleaning_pipeline.to_serialized_data()
                            if self.cleaning_pipeline
                            else None
                        ),
                        chunk,
                        self._code_col,
                        self._description_col,
                        digits,
                    )
                )
                for chunk in chunks
            ]
            # Distribute the serialized tasks to the workers
            serialized_results = executor.map(generate_chunk_wrapper, serialized_tasks)

            # Collect the results
            all_results.extend(serialized_results)
        return all_results

    def _chunk_rows(
        self, data: List[List[str]], max_workers: Optional[int] = None
    ) -> List[List[List[str]]]:
        max_workers = max_workers or 4
        chunk_size = max(1, len(data) // max_workers)
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
        return chunks

    def _merge_results(self, results: list[dict[str, set[str]]]) -> dict[str, set[str]]:
        codes = {}

        for result in results:
            for subheading, descriptions in result.items():
                if subheading in codes:
                    codes[subheading].update(descriptions)
                else:
                    codes[subheading] = descriptions
        return codes

    def _max_workers(self, core_percentage: float = 0.8) -> int:
        cores = os.cpu_count() or 1
        return max(1, int(cores * core_percentage))
