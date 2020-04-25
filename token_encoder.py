import torch
import torch.nn as nn
import numpy as np
import config as Config

class OneHotTokenEncoder(nn.Module):
    def __init__(self, vocab_tokens):
        super(OneHotTokenEncoder, self).__init__()

        self.token2idx = vocab_tokens
        self.emb_dim = len(vocab_tokens)

    def one_hot(self, token_id):

        encoded = np.zeros(len(self.token2idx), dtype=int)
        encoded[token_id] = 1
        return encoded

    def forward(self, samples):

        unk_idx = self.token2idx[Config.UNK_TOKEN]
        pad_idx = self.token2idx[Config.PAD_TOKEN]

        encoded = [[self.token2idx.get(token, unk_idx) for token in tokens] for tokens in samples]

        maxlen = max([len(s) for s in samples])

        padded = np.zeros((len(samples), maxlen), dtype=int)
        masks = torch.zeros(len(samples), maxlen).long()

        for i in range(len(encoded)):
            padded[i, :len(encoded[i])] = np.array(encoded[i])
            masks[i, :len(encoded[i])] = 1

        encoded = [[self.one_hot(token) for token in tokens] for tokens in padded]
        encoded = torch.tensor(encoded).long()

        if torch.cuda.is_available():
            encoded = encoded.cuda()
            masks = masks.cuda()

        result = {
            'mask': masks,
            'encoded': encoded
        }

        return result
