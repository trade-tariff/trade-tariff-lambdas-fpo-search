from training.cleaning_pipeline import Cleaner, debug


class CodeMapper(Cleaner):
    def __init__(self, code_mappings: dict[str, str], digits=8) -> None:
        super().__init__()
        self._code_mappings = code_mappings
        self._digits = digits

        # quickly check that all the keys are the expected length
        for key in code_mappings.keys():
            if len(key) != digits:
                raise Exception(f"Map key '{key}' is not of length {digits}")

    @debug
    def filter(
        self, subheading: str, description: str
    ) -> tuple[str | None, str | None, dict]:
        truncated_code = subheading[: self._digits]

        if truncated_code in self._code_mappings:
            return (self._code_mappings[truncated_code], description, {"updated": True})

        return (subheading, description, {"updated": False})


class Map2024CodesTo2025Codes(CodeMapper):
    """
    This cleaner is responsible for mapping training data using 2024 CN codes to 2025 ones
    """

    def __init__(self) -> None:
        super().__init__(
            {
                "85211020": "85211000",
                "85211095": "85211000",
                "85272120": "85272130",
                "85272152": "85272130",
                "85272159": "85272130",
                "85287111": "85287100",
                "85287115": "85287100",
                "85287119": "85287100",
                "85287191": "85287100",
                "85287199": "85287100",
                "85299065": "85299030",
                "85299097": "85299096",
            }
        )
