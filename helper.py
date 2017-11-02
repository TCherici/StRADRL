import logging
import sys
import os
import datetime

def logger_init(log_dir, id, loglevel='info', redirect_tf=True):

    logger = logging.getLogger('StRADRL')
    if loglevel == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    full_log_path = os.path.join(log_dir,  str(id) + '.app.log')
    print("full log path:" + full_log_path)
    filehandler = logging.FileHandler(full_log_path)
    streamhandler = logging.StreamHandler()
    f = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s:%(name)s:%(lineno)s - %(message)s",
                          "%H:%M:%S")
    filehandler.setFormatter(f)
    streamhandler.setFormatter(f)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info('Start of Application log')

    f = open(full_log_path, 'w')
    sys.stdout = Tee(sys.stdout, f)

    return logger
    
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

def generate_id():
    return datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
