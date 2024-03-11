import logging
from joblib import load

from constants_module import Constants
from chat_util_module import ChatUtil
from chat_service_module import ChatService
from chat_repository_module import ChatRepository
from chat_controller_module import ChatController
from transformers import GPT2LMHeadModel


def run_web_app(test_mode=False, show_full_response_for_debug=False):
    constants = Constants()
    chat_util = ChatUtil(logging.DEBUG, constants)

    gpt2_model = initialize_model(constants)

    target_char_questions_and_answers = load(constants.TARGET_CHAR_PROCESSED_QA_PATH)
    target_char_answers = load(constants.TARGET_CHAR_PROCESSED_ANSWERS_PATH)

    chat_msg_history = []

    chat_repository = ChatRepository(chat_msg_history, target_char_questions_and_answers, target_char_answers,
                                     gpt2_model, chat_util)
    chat_service = ChatService(chat_msg_history, chat_repository, constants, chat_util, show_full_response_for_debug=show_full_response_for_debug)

    chat_controller = ChatController(chat_service, constants, chat_util, test_mode)
    chat_controller.init_conf()

    return chat_controller.run()


def initialize_model(constants):
    gpt_model = GPT2LMHeadModel.from_pretrained(constants.GPT_MODEL_OUTPUT_DIR).to(constants.DEVICE)

    return gpt_model
