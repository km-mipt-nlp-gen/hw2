import sys
import traceback
import logging
import torch


class ChatUtil:
    def __init__(self, logging_lvl, constants):
        self.constants = constants
        self.logger = self.set_up_logger(constants, logging_lvl)

    def set_up_logger(self, constants, log_lvl):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_lvl)

        if not logger.hasHandlers():
            handlers = [
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(constants.LOG_FILE_PATH)
            ]

            for handler in handlers:
                self.set_handler_config(handler, log_lvl, constants.LOG_FORMAT, constants.DATE_FORMAT)
                logger.addHandler(handler)
        return logger

    def set_handler_config(self, handler, log_lvl, log_format, date_format):
        handler.setLevel(log_lvl)
        handler.setFormatter(logging.Formatter(log_format, date_format))

    @staticmethod
    def mean_pool(token_embeds: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
        return pool

    def debug(self, message):
        if not message:
            message = ""
        self.logger.debug(message)

    def info(self, message):
        if not message:
            message = ""
        self.logger.info(message)

    def error(self, message):
        traceback_str = traceback.format_exc()  # Get the traceback as a string
        self.logger.error(f'{message}: {traceback_str}')
