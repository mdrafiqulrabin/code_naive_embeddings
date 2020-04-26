MODE = "train" #train/test

TOKEN_TYPE   = "AllToken" #[OnlyId/OnlyTk/AllToken]/word/hcf
DATASET_NAME = "java-large" # java-large/java-med/java-small
DATASET_TYPE = "Reduced" #Original/Reduced/100P
NUM_TARGET   = 11

TITLE = "mbgru" + "_" + MODE + "_" + TOKEN_TYPE

DATA_PATH    = "/scratch/rabin/token_embedding/data/"
RAW_PATH     = DATA_PATH + "Raw/" + DATASET_TYPE + "/" + DATASET_NAME + "_methods.txt"
TOKEN_PATH   = DATA_PATH + "Token/" + DATASET_TYPE + "/" + DATASET_NAME + "_" + TOKEN_TYPE + ".jsonl"
HCF_PATH     = DATA_PATH + "Handcrafted/" + DATASET_TYPE + "/" + DATASET_NAME + "_hcf.csv"

GLOVE_FILE = DATA_PATH + "Glove/" + "glove.6B.50d.txt"
PAD_TOKEN, UNK_TOKEN = '<PAD>', '<UNK>'
PAD_INDEX, UNK_INDEX = 0, 1

MANUAL_SEED = 42

BATCH_SIZE = 30
EPOCH = 1000
PATIENCE = 10

OUTPUT_DIM = NUM_TARGET
LEARNING_RATE = 1e-2
MOMENTUM = 0.99
HIDDEN_LAYER = 2
HIDDEN_DIM = 128
DROPOUT_RATIO = 0.3

LOG_PATH = TITLE + ".log"
CHECKPOINT = TITLE + ".model"

RESULT_PATH = "/scratch/rabin/token_embedding/results/"
MODEL_PATH  = RESULT_PATH + CHECKPOINT
