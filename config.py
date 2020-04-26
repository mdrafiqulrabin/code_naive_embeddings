TOKEN_TYPE = "hcf" #idf/token/word/hcf
MODE = "train" #train/test

TITLE = "mbgru" + "_" + TOKEN_TYPE
LOG_PATH = TITLE + ".log"

MANUAL_SEED = 42

DATA_PATH = "/scratch/rabin/token_embedding/data/"
DATASET_NAME = "java-large"
DATASET_TYPE = "reduced" #reduced/100P
TOKEN_PATH = DATA_PATH + "Token/" + DATASET_NAME + "_" + str(TOKEN_TYPE == "idf") + ".jsonl"
RAW_PATH = DATA_PATH + DATASET_TYPE + "/" + DATASET_NAME + "_methods.txt"
HCF_PATH = DATA_PATH + "Handcrafted/" + DATASET_NAME + "_" + DATASET_TYPE + ".csv"

GLOVE_FILE = DATA_PATH + "Glove/" + "glove.6B.50d.txt"
PAD_TOKEN, UNK_TOKEN = '<PAD>', '<UNK>'
PAD_INDEX, UNK_INDEX = 0, 1

BATCH_SIZE = 30
EPOCH = 1000
PATIENCE = 10

OUTPUT_DIM = 11
LEARNING_RATE = 1e-2
MOMENTUM = 0.99
HIDDEN_LAYER = 2
HIDDEN_DIM = 16
DROPOUT_RATIO = 0.3

CHECKPOINT = TITLE + ".model"

RESULT_PATH = "/scratch/rabin/token_embedding/results/"
MODEL_PATH  = RESULT_PATH + CHECKPOINT
