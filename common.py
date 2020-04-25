import os, re
import config as Config
import torch

# Save Log Message
def saveLogMsg(msg):
    print(msg)
    with open(Config.LOG_PATH, "a") as log_file:
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

# Remove Comments and Get Words
def get_words(file_path):
    contents = ""
    with open(file_path, 'r') as my_file:
        contents = my_file.read().lower()

    def replacer(match):
        s = match.group(0)
        return " " if s.startswith('/') else s
    comments = r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"'
    pattern = re.compile(comments, re.DOTALL | re.MULTILINE)
    contents = re.sub(pattern, replacer, contents)

    contents = ' '.join(re.split(r'[^a-zA-Z]', contents))
    words = contents.split()

    return words
