import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import config as Config
import common as Common

# Multi-layer Bidirectional GRU
class MulBiGRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, drop_ratio):
        super(MulBiGRULayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim//2, bidirectional=True,
                          num_layers=num_layers, dropout=drop_ratio)

    def forward(self, vectors, mask):
        vectors = vectors.float()
        batch_size = vectors.size(0)
        maxlen = vectors.size(1)
        lengths = mask.sum(-1)

        gru_out, _ = self.gru(vectors)  # (batch, seq_len, hidden_dim)
        assert gru_out.size(0) == batch_size
        assert gru_out.size(1) == maxlen
        assert gru_out.size(2) == self.hidden_dim

        # Separate the directions of the GRU
        gru_out = gru_out.view(batch_size, maxlen, 2, self.hidden_dim//2)

        # Pick up the last hidden state per direction
        fw_last_hn = gru_out[range(batch_size), lengths - 1, 0]  # (batch, hidden//2)
        bw_last_hn = gru_out[range(batch_size), 0, 1]  # (batch, hidden//2)

        last_hn = torch.cat([fw_last_hn, bw_last_hn], dim=1)  # (batch, hidden//2) -> (batch, hidden)

        return {'output': last_hn, 'outputs': gru_out}

# RNNClassifier with Encoder and Extractor
class RNNClassifier(nn.Module):
    def __init__(self, encoder, extractor):
        super(RNNClassifier, self).__init__()
        self.encoder = encoder
        self.extractor = extractor
        self.classifier = nn.Linear(extractor.hidden_dim, Config.OUTPUT_DIM)

    def forward(self, tokens, targets=None):
        encoded = self.encoder(tokens)
        extracted = self.extractor(encoded['encoded'], encoded['mask'])
        f = nn.Softmax(dim=1)
        output = f(self.classifier(extracted['output']))
        return output

# Model Handler for MulBiGRU
class MulBiGRUHandler:

    def __init__(self, dataset, encoder, device):
        self.dataset = dataset
        self.encoder = encoder
        self.device = device

    # Init Model
    def get_model(self):

        # Set the MulBiGRULayer
        model_layer = MulBiGRULayer(self.encoder.emb_dim, hidden_dim=Config.HIDDEN_DIM,
                                    num_layers=Config.HIDDEN_LAYER, drop_ratio=Config.DROPOUT_RATIO)
        Common.saveLogMsg("\nModel Layer = {}".format(model_layer))

        # Set the RNNClassifier
        running_model = RNNClassifier(self.encoder, model_layer)
        if torch.cuda.is_available():
            running_model = running_model.to(self.device)
        Common.saveLogMsg("\nRunning Model = {}".format(running_model))

        return running_model

    # Run Model
    def run(self, running_model):

        # Training the Model
        def train(model, optimizer, shuffled_train_set, batch_size):

            model.train()

            total_loss = 0
            batch_tokens, batch_target = [], []

            random.Random(Config.MANUAL_SEED).shuffle(shuffled_train_set)

            for i in range(len(shuffled_train_set)):

                batch_tokens.append(shuffled_train_set[i]['tokens'])
                batch_target.append([shuffled_train_set[i]['label']])

                if len(batch_tokens) == batch_size or i == len(shuffled_train_set) - 1:

                    optimizer.zero_grad()

                    out = model(batch_tokens)

                    y_pred = None
                    y_target = torch.tensor(batch_target).long()

                    if torch.cuda.is_available():
                        y_pred = out.cuda()
                        loss_function = nn.CrossEntropyLoss()
                        loss_function.to(self.device)
                        loss = loss_function(y_pred, y_target.squeeze())
                    else:
                        y_pred = out.cpu()
                        loss = F.cross_entropy(y_pred, y_target.squeeze())

                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                    batch_tokens, batch_target = [], []

            return model, shuffled_train_set

        # Evaluate the Model
        def evaluate(model, val_set, batch_size):

            model.eval()
            total_loss = 0
            batch_tokens, batch_target = [], []
            predictions, actual = [], []

            with torch.no_grad():
                for i in range(len(val_set)):

                    batch_tokens.append(val_set[i]['tokens'])
                    batch_target.append([val_set[i]['label']])

                    if len(batch_tokens) == batch_size or i == len(val_set) - 1:

                        out = model(batch_tokens)

                        y_pred = None
                        y_target = torch.tensor(batch_target).long()

                        actual.extend(batch_target)
                        if torch.cuda.is_available():
                            y_pred = out.cuda()
                            predictions.extend(torch.argmax(y_pred, dim=1).tolist())
                            loss_function = nn.CrossEntropyLoss()
                            loss_function.to(self.device)
                            loss = loss_function(y_pred, y_target.squeeze())
                        else:
                            y_pred = out.cpu()
                            predictions.extend(np.argmax(y_pred, axis=1).tolist())
                            loss = F.cross_entropy(y_pred, y_target.squeeze())

                        total_loss += loss.item()

                        batch_tokens, batch_target = [], []

            val_f1 = f1_score(actual, predictions, average='weighted')
            val_acc = accuracy_score(actual, predictions)

            return predictions, total_loss / len(val_set), val_f1, val_acc

        # Training of Model
        def training_loop(model, dataset):

            optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM)

            shuffled_train_set = dataset[0]
            best_f1 = 0
            patience_track = 0

            for epoch in range(Config.EPOCH):

                epoch_msg = '[Epoch {}] / {}'.format(epoch + 1, Config.EPOCH)

                model, shuffled_train_set = train(model, optimizer, shuffled_train_set, batch_size=Config.BATCH_SIZE)

                train_pred, train_loss, train_f1, train_acc = evaluate(model, shuffled_train_set, batch_size=Config.BATCH_SIZE)
                epoch_msg += ' [TRAIN] Loss: {:.4f}, F1: {:.4f}, Acc: {:.4f}'.format(train_loss, train_f1, train_acc)
                val_pred, val_loss, val_f1, val_acc = evaluate(model, dataset[1], batch_size=Config.BATCH_SIZE)
                epoch_msg += ' [VAL] Loss: {:.4f}, F1: {:.4f}, Acc: {:.4f}'.format(val_loss, val_f1, val_acc)

                best_f1, epoch_track, patience_track = Common.track_best_model(Config.CHECKPOINT, model, epoch + 1,
                                    best_f1, val_f1, val_acc, val_loss, patience_track)
                Common.saveLogMsg(epoch_msg + epoch_track)
                if patience_track == int(Config.PATIENCE):
                    Common.saveLogMsg('\nNo accuracy improvement for {} consecutive epochs, stopping training!'
                                    .format(Config.PATIENCE))
                    break

            Common.saveLogMsg('Done Training.')

            state = torch.load(Config.CHECKPOINT)
            model.load_state_dict(state['model'])

            Common.saveLogMsg('\nReturning best model - epoch {}, loss {:.4f}, f1-score {:.4f}, accuracy {:.4f}'.
                format(state['epoch'], state['loss'], state['f1'], state['acc']))

            return model

        # Start Training
        assert running_model is not None
        best_model = training_loop(running_model, self.dataset)
        test_pred, test_loss, test_f1, test_acc = evaluate(best_model, self.dataset[2], batch_size=Config.BATCH_SIZE)
        Common.saveLogMsg('\n[Test] Loss: {:.4f}, F1: {:.4f}, Acc: {:.4f}'.format(test_loss, test_f1, test_acc))
        return best_model
