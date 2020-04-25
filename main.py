import config as Config

import numpy as np
np.random.seed(Config.MANUAL_SEED)
import torch
torch.manual_seed(Config.MANUAL_SEED)

# Utilities
log_file = Config.RUN + ".log"
open(log_file, 'w').close()
def saveLogMsg(msg):
    print(msg)
    with open(log_file, "a") as myfile:
        myfile.write(msg + "\n")

# Set Device (cuda/cpu)
saveLogMsg("\nAttaching device...")
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
saveLogMsg("device = {}".format(str(device)))

# Loading Data
saveLogMsg("\nLoading data...")
train_set, val_set, test_set = [], [], []
with open(Config.TOKEN_PATH) as jsons_file:
    for each_json in jsons_file:
        json_dict = eval(str(each_json))
        path = json_dict['filename']
        method = (path.split("_")[-1]).replace(".java", "")
        tokens = list(set(json_dict['tokens']))
        entry = {'path': path, 'method': method, 'tokens': tokens}
        if Config.DATASET_NAME + "/training/" in path:
            train_set.append(entry)
        elif Config.DATASET_NAME + "/validation/" in path:
            val_set.append(entry)
        else:
            test_set.append(entry)
saveLogMsg("#training={}, #validation={}, #test={}".format(len(train_set), len(val_set), len(test_set)))

# Vocabulary of Tokens
saveLogMsg("\nCreating vocabulary...")
train_tokens = [sample['tokens'] for sample in train_set]
flat_train_tokens = sum(train_tokens, [])
val_tokens = [sample['tokens'] for sample in val_set]
flat_val_tokens = sum(val_tokens, [])
vocab_tokens = list(set(flat_train_tokens + flat_val_tokens))
saveLogMsg('Vocabulary size: {}\n'.format(len(vocab_tokens)))