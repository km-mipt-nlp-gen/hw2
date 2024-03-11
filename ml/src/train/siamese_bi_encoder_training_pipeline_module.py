from models_zoo_module import SiameseBiEncoder
from typing import Callable
import plotly.graph_objects as go

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import optuna
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader


class SiameseBiEncoderDataset(Dataset):
    def __init__(self, preprocessed_data, constants, chat_util):
        self.constants = constants
        self.chat_util = chat_util
        tokenized_premises, tokenized_hypothesis = self.tokenize_preprocessed_data(preprocessed_data)
        self.premise_tokens = tokenized_premises
        self.hypothesis_tokens = tokenized_hypothesis
        self.labels = self.get_labels(preprocessed_data)
        self.data = self.init_data()

    def init_data(self):
        data_list = []
        for pt_ids, pt_am, ht_ids, ht_am, label in zip(
                self.premise_tokens["input_ids"], self.premise_tokens["attention_mask"],
                self.hypothesis_tokens["input_ids"], self.hypothesis_tokens["attention_mask"],
                self.labels
        ):
            data = {}
            data["premise_input_ids"] = torch.tensor(pt_ids, dtype=torch.long)
            data["premise_attention_mask"] = torch.tensor(pt_am, dtype=torch.long)
            data["hypothesis_input_ids"] = torch.tensor(ht_ids, dtype=torch.long)
            data["hypothesis_attention_mask"] = torch.tensor(ht_am, dtype=torch.long)
            data["label"] = torch.tensor(label, dtype=torch.long)
            data_list.append(data)
        return data_list

    def __getitem__(self, ix: int):
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)

    def tokenize_preprocessed_data(self, preprocessed_data):
        tokenized_premises = self.constants.TOKENIZER(
            [data[self.constants.PREMISE_UPDATED_COL] for data in preprocessed_data],
            max_length=self.constants.MAX_LENGTH, padding="max_length",
            truncation=True, verbose=True)

        tokenized_hypothesis = self.constants.TOKENIZER(
            [data[self.constants.TARGET_CHAR_ANSWER_COL] for data in preprocessed_data],
            max_length=self.constants.MAX_LENGTH, padding="max_length",
            truncation=True, verbose=True)
        return tokenized_premises, tokenized_hypothesis

    def get_labels(self, preprocessed_data):
        return [data[self.constants.LABEL_COL] for data in preprocessed_data]


class SiameseBiEncoderTrainingPipeline:
    def __init__(self, preprocessed_data, constants, chat_util):
        self.constants = constants
        self.chat_util = chat_util
        self.preprocessed_data = preprocessed_data
        self.siamese_bi_encoder_dataset = SiameseBiEncoderDataset(preprocessed_data, constants, chat_util)
        self.bi_encoder_model = SiameseBiEncoder(constants, chat_util).to(self.constants.DEVICE)

    def train(self, val_interval=1, n_epochs=1, hyperparams_search=False, optuna_n_trials=4,
              hyper_params_search_n_epochs=1,
              hyper_params_search_val_interval=256, user_opt_lr=2e-6, user_scheduler_type='linear'):
        train_ratio = 0.8
        n_total = len(self.siamese_bi_encoder_dataset)
        n_train = int(n_total * train_ratio)
        n_val = n_total - n_train

        train_dataset, val_dataset = random_split(self.siamese_bi_encoder_dataset, [n_train, n_val])

        batch_size = 16
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        total_steps = len(train_dataset) // batch_size
        warmup_steps = int(0.1 * total_steps)

        optimizer, scheduler = self.set_hyperparams(hyperparams_search, total_steps, warmup_steps,
                                                    n_trials=optuna_n_trials, n_epochs=hyper_params_search_n_epochs,
                                                    val_interval=hyper_params_search_val_interval,
                                                    user_opt_lr=user_opt_lr,
                                                    user_scheduler_type=user_scheduler_type)
        self.chat_util.info('Установлены данные гиперпараметров. Начать обучение модели..')

        loss_fn = torch.nn.CrossEntropyLoss()

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

        self.chat_util.info(
            f'Обучение завершено, последний результат функции потерь на валидационном наборе: {all_mean_val_losses_per_val_interval[-1]}')
        self.do_visualization(all_train_batch_losses, all_mean_val_losses_per_val_interval, val_interval)

        return self.bi_encoder_model, all_train_batch_losses, all_mean_val_losses_per_val_interval

    def set_hyperparams(self, hyperparams_search, total_steps, warmup_steps, n_trials=4, n_epochs=1,
                        val_interval=256, user_opt_lr=2e-6, user_scheduler_type='linear'):
        if hyperparams_search:
            best_params = self.do_hyperparam_search(SiameseBiEncoder, n_trials=n_trials, n_epochs=n_epochs,
                                                    val_interval=val_interval)
            self.bi_encoder_model = SiameseBiEncoder(self.constants, self.chat_util).to(self.constants.DEVICE)

            user_opt_lr = best_params['opt_learning_rate']
            optimizer = torch.optim.AdamW(self.bi_encoder_model.parameters(), lr=user_opt_lr)
            scheduler_type = best_params['scheduler_type']
            if scheduler_type == 'linear':
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=total_steps - warmup_steps)
            elif scheduler_type == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=total_steps - warmup_steps)
            else:
                error_msg = "Неподдерживаемый тип scheduler"
                self.chat_util.error(error_msg)
                raise ValueError(error_msg)
        else:
            optimizer = torch.optim.AdamW(self.bi_encoder_model.parameters(), lr=user_opt_lr)
            scheduler = None
            if user_scheduler_type == 'linear':
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=total_steps - warmup_steps)
            elif user_scheduler_type == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=total_steps - warmup_steps)
            else:
                error_msg = "Неподдерживаемый тип scheduler"
                self.chat_util.error(error_msg)
                raise ValueError(error_msg)

        return optimizer, scheduler

    def get_train_step_fn(self, optimizer: torch.optim.Optimizer,
                          scheduler: torch.optim.lr_scheduler.LambdaLR, loss_fn: torch.nn.CrossEntropyLoss
                          ) -> Callable[[torch.tensor, torch.tensor], float]:
        def train_step_fn(x: torch.tensor, y: torch.tensor) -> float:
            self.bi_encoder_model.train()
            yhat = self.bi_encoder_model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            return loss.item()

        return train_step_fn

    def get_val_step_fn(self, loss_fn: torch.nn.CrossEntropyLoss) -> Callable[[torch.tensor, torch.tensor], float]:
        def val_step_fn(x: torch.tensor, y: torch.tensor) -> float:
            self.bi_encoder_model.eval()
            yhat = self.bi_encoder_model(x)
            loss = loss_fn(yhat, y)
            return loss.item()

        return val_step_fn

    def mini_batch_val(self, dataloader, step_fn):
        val_losses = []
        for _, data in enumerate(dataloader):
            loss = step_fn(data, data[self.constants.LABEL_COL].to(self.constants.DEVICE))
            val_losses.append(loss)
        return np.mean(val_losses)

    def mini_batch(self, dataloader, step_fn, val_dataloader, val_step_fn, val_interval=10):
        train_batch_losses = []
        mean_val_losses_per_val_interval = []

        n_steps = len(dataloader)
        for i, data in enumerate(dataloader):
            train_batch_loss = step_fn(data, data[self.constants.LABEL_COL].to(self.constants.DEVICE))
            train_batch_losses.append(train_batch_loss)

            if i % val_interval == 0:
                mean_val_loss = self.mini_batch_val(val_dataloader, val_step_fn)
                mean_val_losses_per_val_interval.append(mean_val_loss)
                self.chat_util.info(f"Training step {i:>5}/{n_steps}, loss = {train_batch_loss: .3f}")
                self.chat_util.info(f"Validation step {i:>5}/{n_steps}, val_loss = {mean_val_loss: .3f}")

        return np.mean(train_batch_losses), train_batch_losses, mean_val_losses_per_val_interval

    def do_visualization(self, all_train_batch_losses, all_mean_val_losses_per_val_interval, validation_interval,
                         sma_losses=True, window_size_train=32, window_size_val=4, title_msg_prefix='Тренировка'):
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
            title=f"{title_msg_prefix} Siamese-Bi-Encoder: Train (per batch) and Validation (per {validation_interval} batches) Losses"
        )

        fig = go.Figure(data=[trace1, trace2], layout=layout)

        fig.update_layout(
            legend_title="Loss Type",
            height=600,
            width=800,
        )

        fig.show()

    def do_hyperparam_search(self, model_init_fn, n_trials=4, n_epochs=1, val_interval=256):
        self.chat_util.info('Старт поиска в пространстве гиперпараметров..')
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, model_init_fn, n_epochs, val_interval), n_trials=n_trials)

        best_params = study.best_trial.params
        best_lr = best_params['opt_learning_rate']
        best_scheduler_type = best_params['scheduler_type']

        self.chat_util.info('Поиск в пространстве гиперпараметров завершен.')
        self.chat_util.info(f"Лучший Learning Rate: {best_lr}")
        self.chat_util.info(f"Лучший Scheduler Type: {best_scheduler_type}")

        return best_params

    def objective(self, trial, model_init_fn, n_epochs, val_interval):
        lr = trial.suggest_loguniform('opt_learning_rate', 2e-6, 2e-5)
        scheduler_type = trial.suggest_categorical('scheduler_type', ['linear', 'cosine'])

        self.chat_util.info(
            f'Поиск гиперпараметров: кросс-валидация для набора параметров "opt_learning_rate"={lr}; "scheduler_type"={scheduler_type}')

        kf = KFold(n_splits=2, shuffle=True)

        all_splits_val_scores = []
        split_number = 1
        for train_index, val_index in kf.split(range(len(self.siamese_bi_encoder_dataset))):
            self.chat_util.info(f'Поиск гиперпараметров: кросс-валидация - обучение на KFold {split_number}')

            self.bi_encoder_model = model_init_fn(self.constants, self.chat_util).to(self.constants.DEVICE)

            train_dataset = Subset(self.siamese_bi_encoder_dataset, train_index)
            val_dataset = Subset(self.siamese_bi_encoder_dataset, val_index)

            best_val_score, all_train_batch_losses, all_mean_val_losses_per_val_interval = self.train_and_evaluate(
                train_dataset, val_dataset, n_epochs=n_epochs,
                val_interval=val_interval, lr=lr, scheduler_type=scheduler_type)

            self.do_visualization(all_train_batch_losses, all_mean_val_losses_per_val_interval, val_interval,
                                  window_size_val=2,
                                  title_msg_prefix=f'Поиск гиперпараметров: кросс-валидация ("opt_learning_rate"={lr}; "scheduler_type"={scheduler_type}, KFold={split_number})')

            all_splits_val_scores.append(best_val_score)
            split_number = split_number + 1

        mean_val_score = np.mean(all_splits_val_scores)
        return mean_val_score

    def train_and_evaluate(self, train_dataset, val_dataset, val_interval=1, n_epochs=1, lr=2e-6,
                           scheduler_type='linear'):
        batch_size = 16
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.bi_encoder_model.parameters(), lr=lr)
        total_steps = len(train_dataset) // batch_size
        warmup_steps = int(0.1 * total_steps)

        scheduler = None
        if scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps - warmup_steps)
        elif scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps - warmup_steps)
        else:
            error_msg = "Неподдерживаемый тип scheduler"
            self.chat_util.error(error_msg)
            raise ValueError(error_msg)

        loss_fn = torch.nn.CrossEntropyLoss()

        train_step_fn = self.get_train_step_fn(optimizer, scheduler, loss_fn)
        val_step_fn = self.get_val_step_fn(loss_fn)

        all_train_batch_losses = []
        all_mean_val_losses_per_val_interval = []

        for epoch in range(1, n_epochs + 1):
            interval = val_interval
            _, train_batch_losses, mean_val_losses_per_val_interval = self.mini_batch(
                train_dataloader, train_step_fn, val_dataloader, val_step_fn, val_interval=interval)
            all_train_batch_losses.extend(train_batch_losses)
            all_mean_val_losses_per_val_interval.extend(mean_val_losses_per_val_interval)

        min_val_loss = min(all_mean_val_losses_per_val_interval)
        return min_val_loss, all_train_batch_losses, all_mean_val_losses_per_val_interval
