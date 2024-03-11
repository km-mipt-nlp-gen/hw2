import re


class ChatService:
    def __init__(self, chat_msg_history, chat_repository, constants, chat_util, show_full_response_for_debug=False):
        self.repository = chat_repository
        self.repository.chat_msg_history = chat_msg_history
        self.constants = constants
        self.chat_util = chat_util
        self.show_full_response_for_debug = show_full_response_for_debug

    @property
    def chat_msg_history(self):
        return self.repository.chat_msg_history

    @chat_msg_history.setter
    def chat_msg_history(self, value):
        self.repository.chat_msg_history = value

    @property
    def gpt2_model(self):
        return self.repository.gpt2_model

    @gpt2_model.setter
    def gpt2_model(self, value):
        self.repository.gpt2_model = value

    @property
    def target_char_answers(self):
        return self.repository.target_char_answers

    @property
    def target_char_questions_and_answers(self):
        return self.repository.target_char_questions_and_answers

    def enrich_query_with_context(self, query, user):
        formatted_query = f"{user}: {query}"

        # добавить последнюю реплику к chat_msg_history
        self.chat_msg_history.append(formatted_query)

        recent_msgs = self.chat_msg_history[-self.constants.LAG_COUNT:]

        updated_query_parts = []
        for i, msg in enumerate(reversed(recent_msgs), start=1):
            updated_query_parts.append(f'R_{i}: "{msg}"')

        updated_query_parts = updated_query_parts[::-1]

        updated_query = "; ".join(updated_query_parts)
        return updated_query

    def get_answer_gpt2_model(self, query, user):
        self.chat_util.debug("get_answer_gpt2_model - старт выполнения")

        self.chat_util.debug(f'Query: {query}')
        self.chat_util.debug(f'User: {user}')
        query = self.enrich_query_with_context(query, user)
        self.chat_util.debug(f'Обогащенный контекстом запрос: {query}')

        answer = self.get_gpt2model_answer_aux(query)
        self.chat_util.debug(f'gpt модели ответ: {answer}')

        # добавить ответ к chat_msg_history
        self.chat_msg_history.append(answer)
        return self.chat_msg_history

    def get_gpt2model_answer_aux(self, sequence):
        input_ids = self.constants.GPT_TOKENIZER.encode(sequence, return_tensors='pt').to(self.constants.DEVICE)

        generated_text_ids = self.gpt2_model.generate(
            input_ids,
            do_sample=True,
            max_length=self.constants.GPT_MAX_ANSWER_LENGTH,
            num_return_sequences=self.constants.GPT_NUM_RETURN_SEQUENCES,
            pad_token_id=self.gpt2_model.config.eos_token_id,
            top_k=self.constants.GPT_TOP_K,
            top_p=self.constants.GPT_TOP_P
        )

        generated_text = self.constants.GPT_TOKENIZER.decode(generated_text_ids[0], skip_special_tokens=True)

        if self.show_full_response_for_debug:
            self.chat_util.debug(f'gpt модели полный ответ (до выделения ответа персонажа): {generated_text}')

        return self.process_gpt2_answer(generated_text)

    def process_gpt2_answer(self, message):
        match = re.search(self.constants.TARGET_CHAR_NAME_PATTERN, message)

        if match:
            return match.group(1)
        else:
            matches = re.findall(r'R_[0-9]+:\s*"([^"]+)"', message)
            if not matches:
                return ""

            last_replica_with_character_name = matches[-1].strip()

            colon_index = last_replica_with_character_name.find(':')
            if colon_index != -1:
                return last_replica_with_character_name[colon_index + 1:].strip()
            else:
                return last_replica_with_character_name

    def clear_chat_msg_history(self):
        self.chat_msg_history = []
