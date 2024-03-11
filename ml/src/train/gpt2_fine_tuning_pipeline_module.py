from models_zoo_module import GPT2
import plotly.graph_objects as go

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback

import torch
from torch.utils.data import Dataset


class GPT2Dataset(Dataset):
    def __init__(self, preprocessed_data, constants, chat_util):
        self.constants = constants
        self.chat_util = chat_util
        self.tokenizer = self.constants.GPT_TOKENIZER

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.examples = []

        for item in preprocessed_data:
            concatenated_text = item["premise_updated_col"] + self.tokenizer.eos_token + item["target_char_answer_col"]
            encoding = self.tokenizer(concatenated_text, truncation=True, max_length=self.constants.GPT_MAX_LENGTH, padding=self.constants.GPT_PADDING, return_tensors="pt")

            self.examples.append({'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0), 'labels': encoding['input_ids'].squeeze(0)})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.losses.append(logs['loss'])


class GPT2TrainingPipeline:
    def __init__(self, preprocessed_data, constants, chat_util):
        self.constants = constants
        self.chat_util = chat_util
        self.preprocessed_data = preprocessed_data
        self.gpt2_dataset = GPT2Dataset(preprocessed_data, constants, chat_util)
        self.gpt2_model = GPT2.from_pretrained_custom(self.constants, self.chat_util).to(self.constants.DEVICE)
        self.training_args = TrainingArguments(
            output_dir=self.constants.GPT_MODEL_OUTPUT_DIR,
            num_train_epochs=self.constants.GPT_TRAIN_EPOCHS,
            per_device_train_batch_size=self.constants.GPT_BATCH_SIZE,
            warmup_steps=len(preprocessed_data) // 10,
            weight_decay=self.constants.GPT_WEIGHT_DECAY,
            logging_dir=self.constants.GPT_LOGGING_DIR,
            save_strategy=self.constants.GPT_SAVE_STRATEGY,
            save_steps=self.constants.GPT_SAVE_STEPS,
            overwrite_output_dir = self.constants.GPT_OVERWRITE_OUTPUT,
            fp16=self.constants.GPT_FP16,
            report_to="none")
        self.metrics_callback = MetricsCallback()
        self.trainer = Trainer(
            model=self.gpt2_model,
            args=self.training_args,
            train_dataset=self.gpt2_dataset,
            callbacks=[self.metrics_callback])

    def train(self):
        torch.cuda.empty_cache()
        self.trainer.train()
        losses = self.metrics_callback.losses

        self.trainer.save_model(self.constants.GPT_MODEL_OUTPUT_DIR)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, mode='lines', name='Training Loss'))
        fig.update_layout(title='Training Loss Over Time', xaxis_title='Step', yaxis_title='Loss')
        fig.show()

        return self.gpt2_model, losses
