from joblib import load


class ChatRepository:
    def __init__(self, chat_msg_history, target_char_qa_pairs, target_char_answers,
                 bi_encoder_model, cross_encoder_model, chat_service_accelerator,
                 chat_util, preprocessed_questions_answers_embeddings_path=None,
                 preprocessed_answers_embeddings_path=None,
                 preprocessed_questions_answers_embeddings_faiss_index_path=None,
                 preprocessed_questions_answers_embeddings_faiss_psa_index_path=None):
        self.chat_util = chat_util
        if chat_service_accelerator:
            self._preprocessed_questions_answers_embeddings = chat_service_accelerator.preprocess_training_data_embeddings(
                target_char_qa_pairs)
            self._preprocessed_answers_embeddings = chat_service_accelerator.preprocess_answers_embeddings(
                target_char_answers)
            self._preprocessed_questions_answers_embeddings_faiss_index = chat_service_accelerator.create_faiss_index(
                self._preprocessed_questions_answers_embeddings)
            self._preprocessed_questions_answers_embeddings_faiss_psa_index = chat_service_accelerator.create_faiss_psa_index(
                self._preprocessed_questions_answers_embeddings)
            self.chat_util.debug('ChatServiceAccelerator успешно предобработал эмбеддинги для кеширования')
        elif (preprocessed_questions_answers_embeddings_path
              and
              preprocessed_answers_embeddings_path
              and
              preprocessed_questions_answers_embeddings_faiss_index_path
              and
              preprocessed_questions_answers_embeddings_faiss_psa_index_path):
            self._preprocessed_questions_answers_embeddings = load(preprocessed_questions_answers_embeddings_path)
            self._preprocessed_answers_embeddings = load(preprocessed_answers_embeddings_path)
            self._preprocessed_questions_answers_embeddings_faiss_index = load(
                preprocessed_questions_answers_embeddings_faiss_index_path)
            self._preprocessed_questions_answers_embeddings_faiss_psa_index = load(
                preprocessed_questions_answers_embeddings_faiss_psa_index_path)
            self.chat_util.debug(
                'Joblib библиотека успешно загрузила эмбеддинги (предобработанные ранее ChatServiceAccelerator) из файловой системы для кеширования')
        else:
            raise ValueError('Нет ChatServiceAccelerator или путей к предобработанным эмбеддингам')
        self._chat_msg_history = chat_msg_history
        self._bi_encoder_model = bi_encoder_model
        self._cross_encoder_model = cross_encoder_model
        self._target_char_preprocessed_answers = target_char_answers
        self._target_char_preprocessed_qa_pairs = target_char_qa_pairs

    @property
    def chat_msg_history(self):
        return self._chat_msg_history

    @chat_msg_history.setter
    def chat_msg_history(self, value):
        self._chat_msg_history = value

    @property
    def preprocessed_questions_answers_embeddings(self):
        return self._preprocessed_questions_answers_embeddings

    @property
    def preprocessed_answers_embeddings(self):
        return self._preprocessed_answers_embeddings

    @property
    def bi_encoder_model(self):
        return self._bi_encoder_model

    @bi_encoder_model.setter
    def bi_encoder_model(self, value):
        self._bi_encoder_model = value

    @property
    def cross_encoder_model(self):
        return self._cross_encoder_model

    @cross_encoder_model.setter
    def cross_encoder_model(self, value):
        self._cross_encoder_model = value

    @property
    def target_char_answers(self):
        return self._target_char_preprocessed_answers

    @property
    def target_char_questions_and_answers(self):
        return self._target_char_preprocessed_qa_pairs

    @property
    def preprocessed_questions_answers_embeddings_faiss_index(self):
        return self._preprocessed_questions_answers_embeddings_faiss_index

    @property
    def preprocessed_questions_answers_embeddings_faiss_psa_index(self):
        return self._preprocessed_questions_answers_embeddings_faiss_psa_index
