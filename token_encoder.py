import torch
import torch.nn as nn
import numpy as np
import config as Config
import os

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

class GloveWordEmbedder(nn.Module):
    def __init__(self, vocab_tokens, glove_file):
        super(GloveWordEmbedder, self).__init__()
        assert os.path.exists(glove_file) and glove_file.endswith('.txt'), glove_file

        self.emb_dim = None

        self.PAD_TOKEN = Config.PAD_TOKEN
        self.UNK_TOKEN = Config.UNK_TOKEN

        idx2word = [self.PAD_TOKEN, self.UNK_TOKEN]
        idx2vect = [None, None]

        with open(glove_file, 'r') as fp:
            for line in fp:
                line = line.split()

                if line[0] not in vocab_tokens:
                    continue

                w = line[0]
                v = np.array([float(value) for value in line[1:]])

                if self.emb_dim is None:
                    self.emb_dim = v.shape[0]

                idx2word.append(w)
                idx2vect.append(v)

        if len(idx2vect) == 2:
            print("No match with glove !!")
            idx2vect[0] = np.zeros(self.sequence_length)
            idx2vect[1] = np.zeros(self.sequence_length)
            self.emb_dim = self.sequence_length
        else:
            idx2vect[0] = np.zeros(self.emb_dim)
            idx2vect[1] = np.mean(idx2vect[2:], axis=0)

        self.embeddings = torch.from_numpy(np.array(idx2vect)).float()
        self.embeddings = nn.Embedding.from_pretrained(self.embeddings, freeze=False)

        self.idx2word = {i: w for i, w in enumerate(idx2word)}
        self.word2idx = {w: i for i, w in self.idx2word.items()}

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
        self.emb_dim = 39 #TODO

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
        unk_idx = self.token2idx[Config.UNK_TOKEN]
        pad_idx = self.token2idx[Config.PAD_TOKEN]

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
