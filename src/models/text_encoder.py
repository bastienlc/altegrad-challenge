import torch
from torch import nn
from transformers import AutoModel


def mean_pooling(embeddings: torch.Tensor, attention_mask: torch.Tensor):
    """Mean Pooling - Take attention mask into account for correct averaging
    source : https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

    model_output: (batch_size, sequence_length, hidden_size)
    attention_mask: (batch_size, sequence_length)
    """
    hidden_size = embeddings.shape[2]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(-1, -1, hidden_size).float()
    )
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.use_sentence_transformer = model_name in [
            "sentence-transformers/all-MiniLM-L6-v2"
        ]
        self.model = AutoModel.from_pretrained(
            model_name,
        )

    def forward(self, input_ids, attention_mask):
        encoded_text = self.model(input_ids, attention_mask=attention_mask)

        if self.use_sentence_transformer:
            # In this case we can only use mean pooling
            return mean_pooling(encoded_text.last_hidden_state, attention_mask)
        else:
            return encoded_text.last_hidden_state[
                :, 0, :
            ]  # This is the CLS token i.e. it is the embedding of the whole sentence. Mean pooling gives similar results.
