import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(name)s [%(levelname)s] %(message)s"
    )