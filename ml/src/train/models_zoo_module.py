import torch
import datasets
from transformers import AutoModel


class SiameseBiEncoder(torch.nn.Module):
    def __init__(self, constants, chat_util):
        super().__init__()
        self.constants = constants
        self.chat_util = chat_util
        self.max_length = self.constants.MAX_LENGTH
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_tokenizer = self.constants.TOKENIZER
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size * 3, 2)

    def forward(self, data: datasets.arrow_dataset.Dataset) -> torch.tensor:
        premise_input_ids = data["premise_input_ids"].to(self.constants.DEVICE)
        premise_attention_mask = data["premise_attention_mask"].to(self.constants.DEVICE)
        hypothesis_input_ids = data["hypothesis_input_ids"].to(self.constants.DEVICE)
        hypothesis_attention_mask = data["hypothesis_attention_mask"].to(self.constants.DEVICE)

        out_premise = self.bert_model(premise_input_ids, premise_attention_mask)
        out_hypothesis = self.bert_model(hypothesis_input_ids, hypothesis_attention_mask)
        premise_embeds = out_premise.last_hidden_state
        hypothesis_embeds = out_hypothesis.last_hidden_state

        pooled_premise_embeds = self.chat_util.mean_pool(premise_embeds, premise_attention_mask)
        pooled_hypotheses_embeds = self.chat_util.mean_pool(hypothesis_embeds, hypothesis_attention_mask)

        embeds = torch.cat([pooled_premise_embeds, pooled_hypotheses_embeds,
                            torch.abs(pooled_premise_embeds - pooled_hypotheses_embeds)],
                           dim=-1)
        return self.linear(embeds)


class CrossEncoder(torch.nn.Module):
    def __init__(self, constants, chat_util):
        super().__init__()
        self.constants = constants
        self.chat_util = chat_util
        self.max_length = self.constants.MAX_LENGTH
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_tokenizer = self.constants.TOKENIZER
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the CLS token's output
        return self.linear(pooled_output)
