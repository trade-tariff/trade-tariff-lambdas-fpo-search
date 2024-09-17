import re
import logging
import csv

from lingua import Language, LanguageDetectorBuilder
from train_args import TrainScriptArgsParser
from typing import Dict, List

training_args = TrainScriptArgsParser()
logger = logging.getLogger("cleaning_pipeline")


def debug(func):
    def wrapper(*args, **kwargs):
        subheading, description, meta = func(*args, **kwargs)

        if subheading is None or description is None:
            logger.debug(
                f"Skipping {func.__name__} with args {args} and kwargs {kwargs}"
            )

        return subheading, description, meta

    return wrapper


class Cleaner:
    def filter(
        self,
        subheading: str,
        description: str,
    ) -> tuple[str | None, str | None, dict]:
        raise NotImplementedError()


class CleaningPipeline:
    """
    This class represents a pipeline of filters that are applied to subheadings and descriptions.

    The pipeline is applied in order, and if any filter returns None for either the subheading or description,
    the pipeline will return None for both (effectively skipping this code/description pair out of the model).

    The pipeline can also return metadata from each filter if the return_meta flag is set to True.
    This helps with debugging and understanding why a particular code/description pair was skipped.

    The pipeline is completely configurable and can be built from a list of chosen filters for a specific use case.

    Setting the log_level to DEBUG will log each filter that is applied to each code/description pair.
    """

    def __init__(self, cleaners: list[Cleaner], return_meta: bool = False) -> None:
        self._filters = cleaners
        self._return_meta = return_meta

    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        all_meta = {}

        for filter in self._filters:
            subheading, description, meta = filter.filter(subheading, description)

            if self._return_meta:
                all_meta[filter.__class__.__name__] = meta

            if subheading is None or description is None:
                return (None, None, all_meta)

        return subheading, description, all_meta

    def to_serialized_data(self) -> list:
        return [
            (
                filter.to_serialized_data()
                if isinstance(filter, LanguageCleaning)
                else filter
            )
            for filter in self._filters
        ]

    @classmethod
    def from_serialized_data(cls, data: list[dict]) -> "CleaningPipeline":
        filters: List[Cleaner] = [
            (
                LanguageCleaning.from_serialized_data(filter_data)
                if isinstance(filter_data, dict)
                else filter_data
            )
            for filter_data in data
        ]
        return cls(filters)


class StripExcessCharacters(Cleaner):
    """
    This cleaner is responsible for stripping excess characters from the subheading and description.

    It will strip the following characters from the right side of the subheading and description:

    - Newline characters
    - Tab characters
    - Carriage return characters
    - Period characters
    - Comma characters
    """

    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        characters_to_rstrip = ["\n", "\t", "\r", ".", ","]
        subheading = subheading.strip()
        description = " ".join(description.split())

        for character in characters_to_rstrip:
            subheading = subheading.rstrip(character)
            description = description.rstrip(character)

        return (subheading, description, {})


class PluralCleaning(Cleaner):
    """
    This cleaner is responsible for cleaning up plural forms of words in the description.

    Other than for specific forms like "size s" (which we preserve) it will convert "women s" to "womens", etc.
    """

    def __init__(self) -> None:
        self._size_s = re.compile(r"\bsize\s+s\b")
        self._plural_s = re.compile(r"\b(\w+)\s+s\b")

    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        description = re.sub(self._size_s, "size_placeholder", description)
        description = re.sub(self._plural_s, r"\1s", description)
        description = re.sub(r"size_placeholder", "size s", description)

        return (subheading, description, {})


class IncorrectPairsRemover(Cleaner):
    """
    This cleaner is responsible for removing incorrect code-description pairs from the data.

    These were identified by the team and are stored in a CSV file with the following columns:
    - Description
    - Skipped Chapter
    - Skipped Commodity

    If the description matches the description in the CSV file, the code will be skipped when
    it matches the skipped code or skipped chapter.
    """

    DESCRIPTION_COLUMN = 0
    SHOULD_BE_CHAPTER_COLUMN = 1
    SHOULDNT_BE_COMMODITY_COLUMN = 2

    def __init__(self, incorrect_code_desc_pairs: Dict[str, Dict[str, str]]) -> None:
        self._incorrect_code_desc_pairs = incorrect_code_desc_pairs

    @debug
    def filter(self, subheading: str, description: str):
        candidate_skips = self._incorrect_code_desc_pairs.get(description.lower())

        if candidate_skips is None:
            return (subheading, description, {})
        else:
            skipped_code = candidate_skips["skipped_code"] or ""
            kept_chapter = candidate_skips["kept_chapter"] or None

            if subheading == skipped_code:
                return (None, None, {"reason": "Incorrect code for description"})
            elif kept_chapter and subheading[:2] != kept_chapter:
                return (None, None, {"reason": "Incorrect chapter for description"})

        return (subheading, description, {})

    @classmethod
    def build(cls, filename: str) -> "IncorrectPairsRemover":
        with open(filename) as f:
            csv_reader = csv.reader(f)
            next(csv_reader)

            return cls(
                {
                    row[cls.DESCRIPTION_COLUMN].lower(): {
                        "skipped_code": row[cls.SHOULDNT_BE_COMMODITY_COLUMN],
                        "kept_chapter": row[cls.SHOULD_BE_CHAPTER_COLUMN],
                    }
                    for row in csv_reader
                }
            )


class RemoveEmptyDescription(Cleaner):
    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        if not description.strip():
            return (None, None, {"reason": "Empty description"})

        return (subheading, description, {})


class RemoveShortDescription(Cleaner):
    def __init__(self, min_length: int | None) -> None:
        super().__init__()
        self._min_length = min_length or 4

    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        if len(description) <= self._min_length:
            return (None, None, {"reason": "Short description"})

        return (subheading, description, {})


class RemoveSubheadingsNotMatchingRegexes(Cleaner):
    """
    This cleaner filters out subheadings that do not match the provided regexes.

    Typically a subheading matches 6/8 digits (e.g. 12345678) and is used as the code.
    """

    def __init__(self, regexes: list[str]) -> None:
        super().__init__()
        self._regexes = regexes

    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        if any(re.search(regex, subheading) for regex in self._regexes):
            return subheading, description, {}

        return (
            None,
            None,
            {"reason": f"Subheading does not match regexes {self._regexes}"},
        )


class RemoveDescriptionsMatchingRegexes(Cleaner):
    """
    This cleaner filters out descriptions that match the provided regexes.
    """

    def __init__(self, regexes: list[str]) -> None:
        super().__init__()
        self._regexes = regexes

    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        if any(re.search(regex, description) for regex in self._regexes):
            return (
                None,
                None,
                {"reason": f"Description matches regexes {self._regexes}"},
            )

        return (subheading, description, {})


class LanguageCleaning(Cleaner):
    """
    This cleaner filters out known non-English descriptions based on the following criteria:
    1. If the description is in the list of exact keeps, keep it
    2. If the description contains a partial keep, keep it
    3. If the description contains a partial skip, skip it
    4. If the language of the description cannot be detected, keep it
    5. If the detected language is not in the preferred languages, skip it
    """

    def __init__(
        self,
        detected_languages: list[str],
        preferred_languages: list[str],
        partial_skips: list[str],
        partial_keeps: list[str],
        exact_keeps: list[str],
    ) -> None:
        super().__init__()
        self._partial_skips = partial_skips
        self._partial_keeps = partial_keeps
        self._exact_keeps = exact_keeps
        self._preferred_languages = [
            getattr(Language, lang) for lang in preferred_languages
        ]
        self._detected_languages = [
            getattr(Language, lang) for lang in detected_languages
        ]

        self._detector = (
            LanguageDetectorBuilder.from_languages(*self._detected_languages)
            .with_minimum_relative_distance(training_args.minimum_relative_distance())
            .build()
        )

    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        language = self._detector.detect_language_of(description)

        if description in self._exact_keeps:
            return (subheading, description, {})

        for partial_keep in self._partial_keeps:
            if partial_keep in description:
                return (subheading, description, {})

        for partial_skip in self._partial_skips:
            if partial_skip in description:
                return (
                    None,
                    None,
                    {"reason": f"Description contains partial skip {partial_skip}"},
                )

        if language is None:
            return (subheading, description, {})

        if language not in self._preferred_languages:
            return (
                None,
                None,
                {"reason": f"Detected language {language} not in preferred languages"},
            )

        return (subheading, description, {})

    @classmethod
    def from_serialized_data(cls, data: dict) -> "LanguageCleaning":
        return cls(
            data["detected_languages"],
            data["preferred_languages"],
            data["partial_skips"],
            data["partial_keeps"],
            data["exact_keeps"],
        )

    def to_serialized_data(self) -> dict:
        return {
            "detected_languages": [lang.name for lang in self._detected_languages],
            "preferred_languages": [lang.name for lang in self._preferred_languages],
            "partial_skips": self._partial_skips,
            "partial_keeps": self._partial_keeps,
            "exact_keeps": self._exact_keeps,
        }


class NegationCleaning(Cleaner):
    """
    This cleaner is responsible for cleaning up negations in the description.

    For example "tomatoes, other than monster beef tomatoes" becomes "tomatoes"

    This helps prevent false positive matches during inference.
    """

    def __init__(
        self, negation_terms: List[str], non_negation_terms: List[str]
    ) -> None:
        super().__init__()
        self._negation_terms = negation_terms
        self._non_negation_terms = non_negation_terms
        self._bracket_negation_regex = re.compile(
            rf"(\(({'|'.join(negation_terms)}).*\))"
        )
        self._full_negation_regex = re.compile(
            rf"(,|-)?\s*({'|'.join(negation_terms)})\s+((?!-).)*"
        )
        self._non_breaking_space = "\u00A0"

    @classmethod
    def build(cls) -> "NegationCleaning":
        return cls(
            [
                "neither",
                "other than",
                "excluding",
                "not",
                "except",
                "excl.",
            ],
            ["with or without"],
        )

    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        # Remove non-breaking spaces
        description = (description or "").lower().replace(self._non_breaking_space, " ")
        # Remove bracketed negations (e.g. "description (excluding this part)" -> "description")
        description = re.sub(self._bracket_negation_regex, "", description)
        # Remove non-bracketed negations (e.g. "description excluding this part" -> "description")
        description = re.sub(self._full_negation_regex, "", description).strip()

        return (subheading, description, {})


class DescriptionLower(Cleaner):
    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        return (subheading, description.lower(), {})


class PhraseRemover(Cleaner):
    """
    Loads a list of phrases from a file and removes them from the description where the cleaner is applied.


    Example where the phrase is "some phrase" a description like "description with some phrase" becomes "description with"
    """
    def __init__(self, phrases: List[str]) -> None:
        self._phrases = phrases

    @classmethod
    def build(cls, filename: str) -> "PhraseRemover":
        with open(filename) as f:
            return cls(f.read().splitlines())

    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        for phrase in self._phrases:
            description = description.replace(phrase, "")

        if not description.strip():
            return (None, None, {"reason": "Empty description"})

        return (subheading, description, {})
