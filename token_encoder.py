import os

import numpy as np
import torch
import torch.nn as nn

import config as cf


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
        unk_idx = self.token2idx[cf.UNK_TOKEN]
        pad_idx = self.token2idx[cf.PAD_TOKEN]

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


class GloVeEmbedding(nn.Module):
    def __init__(self, vocab_tokens, glove_file):
        super(GloVeEmbedding, self).__init__()
        assert os.path.exists(glove_file) and glove_file.endswith('.txt'), glove_file

        self.emb_dim = None

        self.PAD_TOKEN = cf.PAD_TOKEN
        self.UNK_TOKEN = cf.UNK_TOKEN

        idx2tok = [self.PAD_TOKEN, self.UNK_TOKEN]
        idx2vec = [None, None]

        with open(glove_file, 'r') as fp:
            for line in fp:
                line = line.split()
                if line[0] not in vocab_tokens:
                    continue

                tok, vec = '', []
                try:
                    tok = line[0]
                    vec = np.array([float(value) for value in line[1:]])
                except:
                    continue

                if self.emb_dim is None:
                    self.emb_dim = vec.shape[0]

                idx2tok.append(tok)
                idx2vec.append(vec)

        if len(idx2vec) == 2:
            print("No match with glove !!")
            idx2vec[0] = np.zeros(self.sequence_length)
            idx2vec[1] = np.zeros(self.sequence_length)
            self.emb_dim = self.sequence_length
        else:
            idx2vec[0] = np.zeros(self.emb_dim)
            idx2vec[1] = np.mean(idx2vec[2:], axis=0)

        self.embeddings = torch.from_numpy(np.array(idx2vec)).float()
        self.embeddings = nn.Embedding.from_pretrained(self.embeddings, freeze=False)

        self.idx2word = {i: t for i, t in enumerate(idx2tok)}
        self.word2idx = {t: i for i, t in self.idx2word.items()}

    def forward(self, samples):
        pad_idx = self.word2idx[self.PAD_TOKEN]
        unk_idx = self.word2idx[self.UNK_TOKEN]

        maxlen = max([len(s) for s in samples])

        encoded = [[self.word2idx.get(token, unk_idx) for token in tokens] for tokens in samples]

        padded = np.zeros((len(samples), maxlen), dtype=int)
        masks = torch.zeros(len(samples), maxlen).long()

        for i in range(len(encoded)):
            padded[i, :len(encoded[i])] = np.array(encoded[i])
            masks[i, :len(encoded[i])] = 1

        encoded = torch.tensor(padded).long()

        if torch.cuda.is_available():
            encoded = encoded.cuda()
            masks = masks.cuda()

        result = {
            'encoded': self.embeddings(encoded),
            'mask': masks,
        }

        return result


class HandcraftedFeatureEncoder(nn.Module):
    def __init__(self):
        super(HandcraftedFeatureEncoder, self).__init__()
        self.emb_dim = 39  # Check

    def forward(self, samples):
        encoded = np.zeros((len(samples), self.emb_dim), dtype=int)
        for i in range(len(encoded)):
            encoded[i, :] = np.array(samples[i])
        encoded = [[tokens] for tokens in encoded]

        masks = torch.zeros(len(encoded), len(encoded[0])).long()
        for i in range(len(encoded)):
            masks[i, :] = 1

        encoded = torch.tensor(encoded).long()
        lengths = masks.sum(-1)

        if torch.cuda.is_available():
            encoded = encoded.cuda()
            masks = masks.cuda()

        result = {
            'mask': masks,
            'encoded': encoded
        }

        return result


class MethodTokenEncoder(nn.Module):
    def __init__(self, vocab_tokens):
        super(MethodTokenEncoder, self).__init__()
        self.token2idx = vocab_tokens
        self.emb_dim = len(vocab_tokens)

    def forward(self, samples):
        unk_idx = self.token2idx[cf.UNK_TOKEN]
        pad_idx = self.token2idx[cf.PAD_TOKEN]

        encoded = np.zeros((len(samples), self.emb_dim), dtype=int)
        for i in range(len(encoded)):
            tokens = [self.token2idx.get(token, unk_idx) for token in samples[i]]
            for j in tokens:
                encoded[i, j] = 1
        encoded = [[tokens] for tokens in encoded]

        masks = torch.zeros(len(encoded), len(encoded[0])).long()
        for i in range(len(encoded)):
            masks[i, :] = 1

        encoded = torch.tensor(encoded).long()
        lengths = masks.sum(-1)

        if torch.cuda.is_available():
            encoded = encoded.cuda()
            masks = masks.cuda()

        result = {
            'mask': masks,
            'encoded': encoded
        }

        return result
