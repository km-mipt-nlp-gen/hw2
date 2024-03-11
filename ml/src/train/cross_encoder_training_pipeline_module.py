from models_zoo_module import CrossEncoder
import plotly.graph_objects as go

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers.optimization import get_linear_schedule_with_warmup


class CrossEncoderDataset(Dataset):
    def __init__(self, preprocessed_data, constants, chat_util):
        self.constants = constants
        self.chat_util = chat_util
        self.tokenized_texts = self.tokenize_preprocessed_data(preprocessed_data)
        self.labels = self.get_labels(preprocessed_data)

    def __getitem__(self, ix: int):
        return {
            "input_ids": torch.tensor(self.tokenized_texts["input_ids"][ix], dtype=torch.long),
            "attention_mask": torch.tensor(self.tokenized_texts["attention_mask"][ix], dtype=torch.long),
            "labels": torch.tensor(self.labels[ix], dtype=torch.float)  # Use float for regression
        }

    def __len__(self) -> int:
        return len(self.tokenized_texts["input_ids"])

    def tokenize_preprocessed_data(self, preprocessed_data):
        tokenized_texts = self.constants.TOKENIZER(
            [data[self.constants.PREMISE_UPDATED_COL] for data in preprocessed_data],
            [data[self.constants.TARGET_CHAR_ANSWER_COL] for data in
             preprocessed_data],
            max_length=self.constants.MAX_LENGTH, padding="max_length",
            truncation=True, verbose=True)
        return tokenized_texts

    def get_labels(self, preprocessed_data):
        return [data[self.constants.LABEL_COL] for data in preprocessed_data]


class CrossEncoderTrainingPipeline:
    def __init__(self, preprocessed_data, constants, chat_util):
        self.constants = constants
        self.chat_util = chat_util
        self.preprocessed_data = preprocessed_data
        self.cross_encoder_dataset = CrossEncoderDataset(preprocessed_data, constants, chat_util)
        self.cross_encoder_model = CrossEncoder(constants, chat_util).to(self.constants.DEVICE)

    def train(self, val_interval=1, n_epochs=1):
        train_ratio = 0.8
        n_total = len(self.cross_encoder_dataset)
        n_train = int(n_total * train_ratio)
        n_val = n_total - n_train

        train_dataset, val_dataset = random_split(self.cross_encoder_dataset, [n_train, n_val])

        batch_size = 16
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.cross_encoder_model.parameters(), lr=3e-5)
        total_steps = len(train_dataset) // batch_size
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps - warmup_steps)

        loss_fn = torch.nn.MSELoss()

        train_step_fn = self.get_train_step_fn(optimizer, scheduler, loss_fn)
        val_step_fn = self.get_val_step_fn(loss_fn)

        all_mean_train_batch_losses = []
        all_train_batch_losses = []
        all_mean_val_losses_per_val_interval = []

        for epoch in range(1, n_epochs + 1):
            interval = val_interval
            mean_train_batch_loss, train_batch_losses, mean_val_losses_per_val_interval = self.mini_batch(
                train_dataloader, train_step_fn, val_dataloader, val_step_fn, val_interval=interval)
            all_mean_train_batch_losses.append(mean_train_batch_loss)
            all_train_batch_losses.extend(train_batch_losses)
            all_mean_val_losses_per_val_interval.extend(mean_val_losses_per_val_interval)

            self.chat_util.info(
                f"Epoch {epoch}: Mean train loss (per batch) = {mean_train_batch_loss:.4f}, Last Validation Loss (per all val dataset) = {mean_val_losses_per_val_interval[-1]:.4f}")

        self.do_visualization(all_train_batch_losses, all_mean_val_losses_per_val_interval, val_interval)

        return self.cross_encoder_model, all_train_batch_losses, all_mean_val_losses_per_val_interval

    def get_train_step_fn(self, optimizer: torch.optim.Optimizer,
                          scheduler: torch.optim.lr_scheduler.LambdaLR, loss_fn):
        def train_step_fn(batch):
            self.cross_encoder_model.train()
            input_ids = batch['input_ids'].to(self.constants.DEVICE)
            attention_mask = batch['attention_mask'].to(self.constants.DEVICE)
            labels = batch['labels'].to(self.constants.DEVICE)
            optimizer.zero_grad()
            logits = self.cross_encoder_model(input_ids, attention_mask)
            loss = loss_fn(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item()

        return train_step_fn

    def get_val_step_fn(self, loss_fn):
        def val_step_fn(batch):
            self.cross_encoder_model.eval()
            input_ids = batch['input_ids'].to(self.constants.DEVICE)
            attention_mask = batch['attention_mask'].to(self.constants.DEVICE)
            labels = batch['labels'].to(self.constants.DEVICE)
            with torch.no_grad():
                logits = self.cross_encoder_model(input_ids, attention_mask)
            loss = loss_fn(logits.squeeze(-1), labels)
            return loss.item()

        return val_step_fn

    def mini_batch(self, dataloader, step_fn, val_dataloader, val_step_fn, val_interval=10):
        train_batch_losses = []
        mean_val_losses_per_val_interval = []

        n_steps = len(dataloader)
        for i, data in enumerate(dataloader):
            train_batch_loss = step_fn(data)
            train_batch_losses.append(train_batch_loss)

            if i % val_interval == 0:
                mean_val_loss = self.mini_batch_val(val_dataloader, val_step_fn)
                mean_val_losses_per_val_interval.append(mean_val_loss)
                self.chat_util.info(f"Training step {i:>5}/{n_steps}, loss = {train_batch_loss: .3f}")
                self.chat_util.info(f"Validation step {i:>5}/{n_steps}, val_loss = {mean_val_loss: .3f}")

        return np.mean(train_batch_losses), train_batch_losses, mean_val_losses_per_val_interval

    def mini_batch_val(self, dataloader, step_fn):
        val_losses = []
        for _, data in enumerate(dataloader):
            loss = step_fn(data)
            val_losses.append(loss)
        return np.mean(val_losses)

    def do_visualization(self, all_train_batch_losses, all_mean_val_losses_per_val_interval, validation_interval,
                         sma_losses=True, window_size_train=32, window_size_val=4):
        if sma_losses:
            all_train_batch_losses_sma = []
            for i in range(len(all_train_batch_losses) - window_size_train):
                all_train_batch_losses_sma.append(np.mean(all_train_batch_losses[i:i + window_size_train]))

            all_train_batch_losses = all_train_batch_losses_sma

            all_mean_val_losses_per_val_interval_sma = []
            for i in range(len(all_mean_val_losses_per_val_interval) - window_size_val):
                all_mean_val_losses_per_val_interval_sma.append(
                    np.mean(all_mean_val_losses_per_val_interval[i:i + window_size_val]))

            all_mean_val_losses_per_val_interval = all_mean_val_losses_per_val_interval_sma

        x_train = list(range(len(all_train_batch_losses)))
        x_val = [i * validation_interval for i in range(len(all_mean_val_losses_per_val_interval))]

        trace1 = go.Scatter(x=x_train, y=all_train_batch_losses, mode='lines', name='Train batch losses')
        trace2 = go.Scatter(x=x_val, y=all_mean_val_losses_per_val_interval, mode='lines', name='Validation mean loss')

        yaxis_title_train = None
        if sma_losses:
            yaxis_title_train = 'Train Loss (SMA)'
        else:
            yaxis_title_train = 'Train Loss (raw)'

        layout = go.Layout(
            xaxis=dict(title='Batch Number'),
            yaxis=dict(title=yaxis_title_train),
            xaxis2=dict(title='Validation Interval (in Batches)', anchor='y2', overlaying='x', side='top'),
            yaxis2=dict(title='Validation Loss', overlaying='y', side='right'),
            title=f"Cross-Encoder: Train (per batch) and Validation (per {validation_interval} batches) Losses"
        )

        fig = go.Figure(data=[trace1, trace2], layout=layout)

        fig.update_layout(
            legend_title="Loss Type",
            height=600,
            width=800,
        )

        fig.show()
