MODE = "train"  # train/test/debug

TOKEN_TYPE = "char"  # [OnlyId/OnlyToken/AllIdToken]/word/char
ENCODE_TYPE = "onehot"  # onehot/embed/GloVe/program
DATASET_NAME = "java-large"  # java-large/java-med/java-small
DATASET_TYPE = "Cleaned"  # Original/Reduced/Cleaned
NUM_TARGET = 11

TITLE = "mbgru" + "_" + MODE + "_" + TOKEN_TYPE

DATA_PATH = ".../token_embedding/data/"
RAW_PATH = DATA_PATH + "Raw/" + DATASET_TYPE + "/" + DATASET_NAME + "_methods.txt"
TOKEN_PATH = DATA_PATH + "Token/" + DATASET_TYPE + "/" + DATASET_NAME + "_" + TOKEN_TYPE + ".jsonl"

GLOVE_FILE = DATA_PATH + "GloVe/glove.6B.300d.txt"
if TOKEN_TYPE == "char": GLOVE_FILE = DATA_PATH + "GloVe/glove.840B.300d-" + TOKEN_TYPE + ".txt"
PAD_TOKEN, UNK_TOKEN = '<PAD>', '<UNK>'
PAD_INDEX, UNK_INDEX = 0, 1

MANUAL_SEED = 42

BATCH_SIZE = 16
EPOCH = 1000
PATIENCE = 10

OUTPUT_DIM = NUM_TARGET
LEARNING_RATE = 1e-2
MOMENTUM = 0.99
HIDDEN_LAYER = 2
HIDDEN_DIM = 128
DROPOUT_RATIO = 0.3

RESULT_PATH = ".../token_embedding/results/"
MODEL_PATH = RESULT_PATH + TITLE + ".model"
LOG_PATH = RESULT_PATH + TITLE + ".log"
