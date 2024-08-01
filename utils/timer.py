from logging import INFO, Logger
from time import perf_counter
from aws_lambda_powertools import Logger as AwsLogger


class CodeTimerFactory:
    def __init__(self, logger: Logger | AwsLogger) -> None:
        self._logger = logger

    def time_code(self, description: str):
        return CodeTimer(description=description, logger=self._logger)


class CodeTimer:
    def __init__(self, description: str, logger: Logger | AwsLogger, level: int = INFO) -> None:
        self._description = description
        self._logger = logger
        self._level = level

    def __enter__(self):
        self._start_time = perf_counter()

    def __exit__(self, *args):
        execution_time_ms = (perf_counter() - self._start_time) * 1000
        msg = f"⏱️  {self._description} took {round(execution_time_ms, 2)}ms"
        self._logger.info(msg=msg, extra={"description" "exec_time_ms": execution_time_ms})
