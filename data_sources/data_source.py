from typing import List


class DataSource:
    def get_codes(self, digits: int) -> dict[str, List[str]]:
        raise NotImplementedError()

    def get_description(self) -> str:
        raise NotImplementedError()
