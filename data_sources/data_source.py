from typing import List


class DataSource:
    def __init__(
        self,
        description: str,
        authoritative: bool = False,
        creates_codes=False,
        multiplier: int = 1,
    ) -> None:
        self.description = description
        self.authoritative = authoritative
        self.creates_codes = creates_codes
        self.multiplier = multiplier

    def get_codes(self, digits: int) -> dict[str, List[str]]:
        raise NotImplementedError()
