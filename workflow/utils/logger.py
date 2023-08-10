import logging

class ColorFormatter(logging.Formatter):

    green= "\x1b[0;42m"
    blue = "\x1b[0;44m"
    yellow = "\x1b[0;43m"
    red = "\x1b[0;41m"
    purple = "\x1b[0;45m"
    reset = "\x1b[0m"
    format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s | (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: purple + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(ColorFormatter())
logger.addHandler(ch)