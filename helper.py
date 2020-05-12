import os, re
import config as cf
import torch

# Save Log Message
def saveLogMsg(msg):
    print(msg)
    with open(cf.LOG_PATH, "a") as log_file:
        log_file.write(msg + "\n")

# Track Best Model
def track_best_model(path, model, epoch, best_f1, val_f1, val_acc, val_loss, patience):
    if best_f1 >= val_f1:
        return best_f1, '', patience + 1
    state = {
        'epoch': epoch,
        'f1': val_f1,
        'acc': val_acc,
        'loss': val_loss,
        'model': model.state_dict()
    }
    torch.save(state, path)
    return val_f1, ' *** ', 0

# Target List
TargetList = ["equals", "main", "setUp", "onCreate", "toString", "run", "hashCode", "init", "execute", "get", "close"]

# Remove Comments and Get Words/Chars
def get_chars_or_words(file_path):
    contents = ""
    with open(file_path, 'r') as my_file:
        contents = my_file.read()

    def replacer(match):
        s = match.group(0)
        return " " if s.startswith('/') else s
    comments = r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"'
    pattern = re.compile(comments, re.DOTALL | re.MULTILINE)
    contents = re.sub(pattern, replacer, contents)

    if cf.TOKEN_TYPE == "word":
        contents = ' '.join(re.split(r'[^a-zA-Z]', contents))
        contents = contents.lower().split()
    elif cf.TOKEN_TYPE == "char":
        contents = list(contents)

    return contents

# vocab2idx and idx2vocab
def get_vocab2idx_idx2vocab(vocab_tokens):
    vocab2idx = {w: i + 2 for i, w in enumerate(vocab_tokens)}
    idx2vocab = {i + 2: w for i, w in enumerate(vocab_tokens)}
    vocab2idx[cf.PAD_TOKEN], vocab2idx[cf.UNK_TOKEN] = cf.PAD_INDEX, cf.UNK_INDEX
    idx2vocab[cf.PAD_INDEX], idx2vocab[cf.UNK_INDEX] = cf.PAD_TOKEN, cf.UNK_TOKEN
    saveLogMsg('vocab2idx size: {}'.format(len(vocab2idx)))
    saveLogMsg('idx2vocab size: {}'.format(len(idx2vocab)))
    return vocab2idx, idx2vocab
