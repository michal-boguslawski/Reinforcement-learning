import logging

class NoDictFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not isinstance(record.msg, dict)
