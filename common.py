import config as Config

def saveLogMsg(msg):
    print(msg)
    with open(Config.LOG_PATH, "a") as log_file:
        log_file.write(msg + "\n")
