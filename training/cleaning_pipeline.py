import re
import logging

from lingua import Language, LanguageDetectorBuilder
from train_args import TrainScriptArgsParser
from typing import List

training_args = TrainScriptArgsParser()
logger = logging.getLogger("cleaning_pipeline")


def debug(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        if result is None:
            logger.debug(
                f"Skipping {func.__name__} with args {args} and kwargs {kwargs}"
            )

        return result

    return wrapper


class Cleaner:
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        raise NotImplementedError()


class CleaningPipeline:
    def __init__(self, filters: list[Cleaner]) -> None:
        self._filters = filters

    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        data = (subheading, description)

        for filter in self._filters:
            data = filter.filter(subheading, description)

            if data is None:
                return None

        return data

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


class StripExcessWhitespace(Cleaner):
    @debug
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        return (subheading.strip(), " ".join(description.split()))


class RemoveEmptyDescription(Cleaner):
    @debug
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        if not description.strip():
            return None

        return (subheading, description)


class RemoveShortDescription(Cleaner):
    def __init__(self, min_length: int | None) -> None:
        super().__init__()
        self._min_length = min_length or 4

    @debug
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        if len(description) <= self._min_length:
            return None

        return (subheading, description)


class RemoveSubheadingsNotMatchingRegexes(Cleaner):
    def __init__(self, regexes: list[str]) -> None:
        super().__init__()
        self._regexes = regexes

    @debug
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        if any(re.search(regex, subheading) for regex in self._regexes):
            return subheading, description

        return None


class RemoveDescriptionsMatchingRegexes(Cleaner):
    def __init__(self, regexes: list[str]) -> None:
        super().__init__()
        self._regexes = regexes

    @debug
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        if any(re.search(regex, description) for regex in self._regexes):
            return None

        return (subheading, description)


class LanguageCleaning(Cleaner):
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
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        """
        This method filters out known non-English descriptions based on the following criteria:
        1. If the description is in the list of exact keeps, keep it
        2. If the description contains a partial keep, keep it
        3. If the description contains a partial skip, skip it
        4. If the language of the description cannot be detected, keep it
        5. If the detected language is not in the preferred languages, skip it
        """

        language = self._detector.detect_language_of(description)

        if description in self._exact_keeps:
            return (subheading, description)

        for partial_keep in self._partial_keeps:
            if partial_keep in description:
                return (subheading, description)

        for partial_skip in self._partial_skips:
            if partial_skip in description:
                return None

        if language is None:
            return (subheading, description)

        if language not in self._preferred_languages:
            return None

        return (subheading, description)

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
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        # Remove non-breaking spaces
        description = (description or "").lower().replace(self._non_breaking_space, " ")
        # Remove bracketed negations (e.g. "description (excluding this part)" -> "description")
        description = re.sub(self._bracket_negation_regex, "", description)
        # Remove non-bracketed negations (e.g. "description excluding this part" -> "description")
        description = re.sub(self._full_negation_regex, "", description).strip()

        return (subheading, description)

class DescriptionLower(Cleaner):
    @debug
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        return (subheading, description.lower())
