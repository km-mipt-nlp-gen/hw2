from joblib import dump, load
import numpy as np
import torch
import faiss
from sklearn.decomposition import PCA


class ChatServiceAccelerator:
    def __init__(self, bi_encoder_model, cross_encoder_model, target_char_questions_and_answers, target_char_answers,
                 constants, chat_util):
        self.model = bi_encoder_model
        self.cross_encoder_model = cross_encoder_model
        self.target_char_questions_and_answers = target_char_questions_and_answers
        self.target_char_answers = target_char_answers
        self.constants = constants
        self.chat_util = chat_util

    def preprocess_answers_embeddings(self, answers, path=None):
        precomputed_embeddings = []
        for answer in answers:
            precomputed_embeddings.append(self.preprocess_answer_embedding(answer))
        precomputed_embeddings = np.array(precomputed_embeddings).squeeze(1)

        if path:
            dump(precomputed_embeddings, path)
        return precomputed_embeddings

    def preprocess_answer_embedding(self, answer):
        self.ensure_model_on_device()
        answer_tokens = self.model.bert_tokenizer(answer, return_tensors="pt", padding='max_length', truncation=True,
                                                  max_length=128).to(self.constants.DEVICE)
        with torch.no_grad():
            answer_embeds = self.model.bert_model(input_ids=answer_tokens['input_ids'],
                                                  attention_mask=answer_tokens['attention_mask']).last_hidden_state
            pooled_answer_embeds = self.chat_util.mean_pool(answer_embeds, answer_tokens['attention_mask'])
        return pooled_answer_embeds.cpu().numpy()

    def ensure_model_on_device(self):
        current_device = next(self.model.parameters()).device
        target_device = self.constants.DEVICE
        if current_device != target_device:
            self.model = self.model.to(target_device)

    def preprocess_training_data_embeddings(self, questions_answers, path=None):
        self.ensure_model_on_device()
        training_data_embeddings = []

        for qa_pair in questions_answers:
            question, answer = qa_pair[self.constants.PREMISE_UPDATED_COL], qa_pair[
                self.constants.TARGET_CHAR_ANSWER_COL]

            question_tokens = self.model.bert_tokenizer(question, return_tensors="pt", padding='max_length',
                                                        truncation=True,
                                                        max_length=128).to(self.constants.DEVICE)
            answer_tokens = self.model.bert_tokenizer(answer, return_tensors="pt", padding='max_length',
                                                      truncation=True,
                                                      max_length=128).to(self.constants.DEVICE)

            with torch.no_grad():
                question_embeds = self.model.bert_model(input_ids=question_tokens['input_ids'],
                                                        attention_mask=question_tokens[
                                                            'attention_mask']).last_hidden_state
                answer_embeds = self.model.bert_model(input_ids=answer_tokens['input_ids'],
                                                      attention_mask=answer_tokens['attention_mask']).last_hidden_state

                pooled_question_embeds = self.chat_util.mean_pool(question_embeds, question_tokens['attention_mask'])
                pooled_answer_embeds = self.chat_util.mean_pool(answer_embeds, answer_tokens['attention_mask'])

                embeds = torch.cat([pooled_question_embeds, pooled_answer_embeds,
                                    torch.abs(pooled_question_embeds - pooled_answer_embeds)], dim=-1)

                training_data_embeddings.append(embeds.cpu().numpy())
        training_data_embeddings = np.array(training_data_embeddings).squeeze(1)

        if path:
            dump(training_data_embeddings, path)
        return training_data_embeddings

    def create_faiss_index(self, preprocessed_question_answer_embeddings, gpu_index=None, path=None):
        if gpu_index is None:
            gpu_index = self.constants.GPU_FAISS_INDEX

        d = preprocessed_question_answer_embeddings.shape[1]

        if (faiss.get_num_gpus() > 0) and gpu_index:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, d)
        else:
            index = faiss.IndexFlatL2(d)
        index.add(preprocessed_question_answer_embeddings.astype('float32'))

        if path:
            dump(index, path)
        return index

    def apply_pca_psa(self, embeddings, n_components=None):
        if n_components is None:
            n_components = self.constants.PCA_COMPONENTS_COUNT
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        return reduced_embeddings

    def create_faiss_psa_index(self, preprocessed_question_answer_embeddings, gpu_index=None,
                               n_components=None, path=None):
        if gpu_index is None:
            gpu_index = self.constants.GPU_FAISS_INDEX

        if n_components is None:
            n_components = self.constants.PCA_COMPONENTS_COUNT

        reduced_embeddings = self.apply_pca_psa(preprocessed_question_answer_embeddings, n_components=n_components)
        d = reduced_embeddings.shape[1]

        if faiss.get_num_gpus() > 0 and gpu_index:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, d)
        else:
            index = faiss.IndexFlatL2(d)
        index.add(reduced_embeddings.astype('float32'))

        if path:
            dump(index, path)
        return index
