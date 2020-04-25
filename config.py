RUN = "mbgru_token"
MANUAL_SEED = 42

LOG_PATH = RUN + ".log"
DATA_PATH = "/scratch/rabin/token_embedding/data/"
DATASET_NAME = "java-large"
TOKEN_PATH = DATA_PATH + "Token/" + DATASET_NAME + ".jsonl"

PAD_TOKEN, UNK_TOKEN = '<PAD>', '<UNK>'
PAD_INDEX, UNK_INDEX = 0, 1
