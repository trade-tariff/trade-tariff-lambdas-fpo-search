class DataSource:
    def __init__(
        self,
        description: str,
        authoritative: bool = False,
        creates_codes: bool = False,
        multiplier: int = 1,
        cleaning_pipeline=None,
    ) -> None:
        self.description = description
        self.authoritative = authoritative
        self.creates_codes = creates_codes
        self.multiplier = multiplier
        self.cleaning_pipeline = cleaning_pipeline

    def get_codes(self, digits: int) -> dict[str, set[str]]:
        raise NotImplementedError()
