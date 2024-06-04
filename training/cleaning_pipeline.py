import re
import logging

from lingua import Language, LanguageDetectorBuilder
from train_args import TrainScriptArgsParser

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
        skip: list[str],
        keep: list[str],
    ) -> None:
        super().__init__()
        self._skip = skip
        self._keep = keep
        self._preferred_languages = [
            getattr(Language, lang) for lang in preferred_languages
        ]

        self._detector = (
            LanguageDetectorBuilder.from_languages(
                *[getattr(Language, lang) for lang in detected_languages]
            )
            .with_minimum_relative_distance(training_args.minimum_relative_distance())
            .build()
        )

    @debug
    def filter(self, subheading: str, description: str) -> tuple[str, str] | None:
        language = self._detector.detect_language_of(description)

        if description in self._skip:  # Skip what would be false positives
            return None

        if description in self._keep:  # Preserve what would be false negatives
            return (subheading, description)

        if language is None:
            return (subheading, description)

        if language not in self._preferred_languages:
            return None

        return (subheading, description)
