import os
import sys
if sys.path[-1] != os.getcwd():
    sys.path.append(os.getcwd())

import time
import json

from log.basic_logger import BasicLogger
from config.config_dict import Config

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

class TrainLogger(BasicLogger):
    def __init__(self, args, config, create=True):
        self.args = args

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if args.get('mark') == None:
            savetag = timestamp + '_' + args.get('model') + '_' + 'repeat' + str(args.get('repeat'))
        else:
            savetag = timestamp + '_' + args.get('model') + '_' + 'repeat' + str(args.get('repeat')) + '_' + args.get('mark')

        save_dir = args.get('save_dir')
        if save_dir == None:
            raise Exception('save_dir can not be None!')
        train_save_dir = os.path.join(save_dir, savetag)
        self.log_dir = os.path.join(train_save_dir, 'log', 'train')
        self.model_dir = os.path.join(train_save_dir, 'model')
        self.result_dir = os.path.join(train_save_dir, 'result')

        if create:
            create_dir([self.log_dir, self.model_dir, self.result_dir])
            print(self.log_dir)
            log_path = os.path.join(self.log_dir, 'Train.log')
            super().__init__(log_path)
            self.record_config(config)

    def record_config(self, config):
        with open(os.path.join(self.log_dir, f'{config}.json'), 'w') as f:
            f.write(json.dumps(self.args))

    def get_log_dir(self):
        if hasattr(self, 'log_dir'):
            return self.log_dir
        else:
            return None

    def get_model_dir(self):
        if hasattr(self, 'model_dir'):
            return self.model_dir
        else:
            return None

    def get_result_dir(self):
        if hasattr(self, 'result_dir'):
            return self.result_dir
        else:
            return None



if __name__ == "__main__":
    args = Config().get_config()
    logger = TrainLogger(args)
    logger.record_config()
    model_path = logger.get_model_dir()
    print(model_path)