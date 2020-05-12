import os, pathlib

import config as cf
import helper as hp

import numpy as np
np.random.seed(cf.MANUAL_SEED)
import torch
torch.manual_seed(cf.MANUAL_SEED)

from token_encoder import *
from model_handler import MulBiGRUHandler

# Create Log File
pathlib.Path(cf.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
open(cf.LOG_PATH, 'w').close()

# Set Device (cuda/cpu)
hp.saveLogMsg("\nAttaching device...")
device = None
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
hp.saveLogMsg("device = {}".format(str(device)))

# Loading Data
hp.saveLogMsg("\nLoading data [{}]...".format(cf.TOKEN_TYPE))
train_set, val_set, test_set = [], [], []

def add_dataset_entry(path, tokens):
    method = (path.split("_")[-1]).replace(".java", "")
    label = hp.TargetList.index(method)
    entry = {'path': path, 'method': method, 'tokens': tokens, 'label': label}
    if cf.DATASET_NAME + "/training/" in path:
        train_set.append(entry)
    elif cf.DATASET_NAME + "/validation/" in path:
        val_set.append(entry)
    else:
        test_set.append(entry)

if cf.TOKEN_TYPE in ["OnlyId", "OnlyTk", "AllToken"]:
    with open(cf.TOKEN_PATH, 'r') as jsons_file:
        for each_json in jsons_file:
            json_dict = eval(str(each_json))
            path = json_dict['filename']
            tokens = list(set(json_dict['tokens']))
            add_dataset_entry(path, tokens)
elif cf.TOKEN_TYPE in ["word", "char"]:
    with open(cf.RAW_PATH, 'r') as path_file:
        for each_path in path_file:
            each_path = each_path.replace('\n', '')
            path = cf.DATA_PATH + "Raw/" + each_path
            tokens = list(hp.get_chars_or_words(path))
            add_dataset_entry(path, tokens)
elif cf.TOKEN_TYPE == "hcf":
    with open(cf.HCF_PATH, 'r') as hcf_file:
        for each_hcf in hcf_file:
            each_hcf = each_hcf.replace('\n', '')
            path = each_hcf.split(',')[0]
            tokens = each_hcf.split(',')[2:]
            tokens = list(map(int, tokens))
            tokens = [1 if token > 0 else 0 for token in tokens]
            add_dataset_entry(path, tokens)

hp.saveLogMsg("#training={}, #validation={}, #test={}".format(len(train_set), len(val_set), len(test_set)))

# Vocabulary of Tokens
vocab_tokens = None
if cf.TOKEN_TYPE != "hcf":
    hp.saveLogMsg("\nCreating vocabulary...")
    train_tokens = [sample['tokens'] for sample in train_set]
    flat_train_tokens = sum(train_tokens, [])
    val_tokens = [sample['tokens'] for sample in val_set]
    flat_val_tokens = sum(val_tokens, [])
    vocab_tokens = list(set(flat_train_tokens + flat_val_tokens))
    hp.saveLogMsg('Vocabulary size: {}'.format(len(vocab_tokens)))

# Token Encoder
encoder = None
if cf.TOKEN_TYPE in ["OnlyId", "OnlyTk", "AllToken"]:
    vocab2idx, idx2vocab = hp.get_vocab2idx_idx2vocab(vocab_tokens)
    encoder = MethodTokenEncoder(vocab2idx)
elif cf.TOKEN_TYPE in ["word", "char"]:
    hp.saveLogMsg("\nLoading GloVe from {}...".format(cf.GLOVE_FILE))
    encoder = GloVeEmbedder(vocab_tokens, cf.GLOVE_FILE)
elif cf.TOKEN_TYPE == "hcf":
    encoder = HandcraftedFeatureEncoder()
else:
    vocab2idx, idx2vocab = hp.get_vocab2idx_idx2vocab(vocab_tokens)
    encoder = OneHotTokenEncoder(vocab2idx)

hp.saveLogMsg("\nInitialized Token Encoder: {}".format(encoder))

# Run Model
data_set = [train_set, val_set, test_set]
handler = MulBiGRUHandler(data_set, encoder, device)
model = handler.get_model()
if cf.MODE == "test" and os.path.exists(cf.MODEL_PATH):
    state = torch.load(cf.MODEL_PATH)
    model.load_state_dict(state['model'])
model = handler.run(model)
