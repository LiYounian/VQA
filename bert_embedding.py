import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import config
from pytorch_transformers import BertModel


class TextProcessor(nn.Module):
    def __init__(self, classes, embedding_features, lstm_features, drop=0.0, use_hidden=True, use_tanh=False,
                 only_embed=False):
        super(TextProcessor, self).__init__()

        # Load pretrained model
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.use_hidden = use_hidden  # return last layer hidden, else return all the outputs for each words
        self.use_tanh = use_tanh
        self.only_embed = only_embed

        self.drop = nn.Dropout(drop)

        if self.use_tanh:
            self.tanh = nn.Tanh()

        if not self.only_embed:
            self.lstm = nn.GRU(input_size=embedding_features,
                               hidden_size=lstm_features,
                               num_layers=1,
                               batch_first=not use_hidden, )

    def forward(self, q, q_len):
        # Predict hidden states features for each layer
        self.bert_model.eval()
        # embedded = self.bert_model(q)

        with torch.no_grad():
            embedded = self.bert_model(q)
        embedded = embedded[0]

        embedded = self.drop(embedded)

        if self.use_tanh:
            embedded = self.tanh(embedded)

        if self.only_embed:
            return embedded

        self.lstm.flatten_parameters()
        if self.use_hidden:
            packed = pack_padded_sequence(embedded, q_len, batch_first=True)
            _, hid = self.lstm(packed)
            return hid.squeeze(0)
        else:
            out, _ = self.lstm(embedded)
            return out
