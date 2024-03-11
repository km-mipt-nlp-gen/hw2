from transformers import GPT2LMHeadModel


class GPT2(GPT2LMHeadModel):
    @classmethod
    def from_pretrained_custom(cls, constants, chat_util, *model_args, **kwargs):
        model = super().from_pretrained(constants.GPT_MODEL_NAME, *model_args, **kwargs)
        return model
