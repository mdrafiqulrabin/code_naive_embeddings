import os

import config as Config
import common as Common

import numpy as np
np.random.seed(Config.MANUAL_SEED)
import torch
torch.manual_seed(Config.MANUAL_SEED)

from token_encoder import OneHotTokenEncoder
from model_handler import MulBiGRUHandler

# Create Log File
open(Config.LOG_PATH, 'w').close()

# Set Device (cuda/cpu)
Common.saveLogMsg("\nAttaching device...")
device = None
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
Common.saveLogMsg("device = {}".format(str(device)))

# Loading Data
Common.saveLogMsg("\nLoading data...")
train_set, val_set, test_set = [], [], []
with open(Config.TOKEN_PATH) as jsons_file:
    for each_json in jsons_file:
        json_dict = eval(str(each_json))
        path = json_dict['filename']
        method = (path.split("_")[-1]).replace(".java", "")
        label = Common.TargetList.index(method)
        tokens = list(set(json_dict['tokens']))
        entry = {'path': path, 'method': method, 'tokens': tokens, 'label': label}
        if Config.DATASET_NAME + "/training/" in path:
            train_set.append(entry)
        elif Config.DATASET_NAME + "/validation/" in path:
            val_set.append(entry)
        else:
            test_set.append(entry)
Common.saveLogMsg("#training={}, #validation={}, #test={}".format(len(train_set), len(val_set), len(test_set)))

# Vocabulary of Tokens
Common.saveLogMsg("\nCreating vocabulary...")
train_tokens = [sample['tokens'] for sample in train_set]
flat_train_tokens = sum(train_tokens, [])
val_tokens = [sample['tokens'] for sample in val_set]
flat_val_tokens = sum(val_tokens, [])
vocab_tokens = list(set(flat_train_tokens + flat_val_tokens))
Common.saveLogMsg('Vocabulary size: {}'.format(len(vocab_tokens)))

# vocab2idx and idx2vocab
vocab2idx = {w:i+2 for i, w in enumerate(vocab_tokens)}
idx2vocab = {i+2:w for i, w in enumerate(vocab_tokens)}
vocab2idx[Config.PAD_TOKEN], vocab2idx[Config.UNK_TOKEN] = Config.PAD_INDEX, Config.UNK_INDEX
idx2vocab[Config.PAD_INDEX], idx2vocab[Config.UNK_INDEX] = Config.PAD_TOKEN, Config.UNK_TOKEN
Common.saveLogMsg('vocab2idx size: {}'.format(len(vocab2idx)))
Common.saveLogMsg('idx2vocab size: {}'.format(len(idx2vocab)))

# OneHot Encoder for Tokens
onehot_encoder = OneHotTokenEncoder(vocab2idx)
Common.saveLogMsg("\nInitialized Token Encoder.")

# Run Model
data_set = [train_set, val_set, test_set]
handler = MulBiGRUHandler(data_set, onehot_encoder, device)
model = handler.get_model()
if os.path.exists(Config.MODEL_PATH):
    state = torch.load(Config.MODEL_PATH)
    model.load_state_dict(state['model'])
model = handler.run(model)
