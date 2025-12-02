import logging
import sys
import os


def setup_logging(path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs debug and higher level messages.
    fh = logging.FileHandler(os.path.join(path, "mcs_tracking.log"), mode="w")
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handler for uncaught exceptions.
    """
    # If the exception is a KeyboardInterrupt, call the default handler
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger(__name__).error(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


# Set the exception hook to our handler
sys.excepthook = handle_exception
