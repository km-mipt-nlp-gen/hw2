class ChatRepository:
    def __init__(self, chat_msg_history, target_char_qa_pairs, target_char_answers, gpt2_fine_tuned_model,
                 chat_util):
        self.chat_util = chat_util
        self._chat_msg_history = chat_msg_history
        self._gpt2_model = gpt2_fine_tuned_model
        self._target_char_preprocessed_answers = target_char_answers
        self._target_char_preprocessed_qa_pairs = target_char_qa_pairs

    @property
    def chat_msg_history(self):
        return self._chat_msg_history

    @chat_msg_history.setter
    def chat_msg_history(self, value):
        self._chat_msg_history = value

    @property
    def gpt2_model(self):
        return self._gpt2_model

    @gpt2_model.setter
    def gpt2_model(self, value):
        self._gpt2_model = value

    @property
    def target_char_answers(self):
        return self._target_char_preprocessed_answers

    @property
    def target_char_questions_and_answers(self):
        return self._target_char_preprocessed_qa_pairs
