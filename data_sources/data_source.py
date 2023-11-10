class DataSource:
    def get_codes(self, digits: int) -> dict[str, list[str]]:
        raise NotImplementedError()
    
    def get_description(self) -> str:
        raise NotImplementedError()
