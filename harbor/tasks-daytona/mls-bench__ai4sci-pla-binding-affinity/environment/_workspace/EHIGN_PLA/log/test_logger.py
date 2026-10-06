import os
import sys
if sys.path[-1] != os.getcwd():
    sys.path.append(os.getcwd())

import time
import json
from glob import glob
import shutil

from log.basic_logger import BasicLogger
from config.config_dict import Config

'''
--model_name
    --model
    --log
        --train
        --test
    --result
        --image
        --pred
        --mask
'''
def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)


class TestLogger(BasicLogger):
    def __init__(self, args):
        self.args = args 

        load_dir = args.get('load_dir')
        if load_dir == None:
            raise Exception('load_dir can not be None!')
        self.model_path = args.get('model_path')
        if self.model_path == None:
            self.model_path = glob(os.path.join(load_dir, 'model', "*"))[-1]
        else:
            self.model_path = os.path.join(load_dir, "model", self.model_path)
        
        self.log_dir = os.path.join(load_dir, 'log', 'test')
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        create_dir([self.log_dir])

        self.result_dir = os.path.join(load_dir, 'result')
        create_dir([self.result_dir])

        log_path = os.path.join(self.log_dir, 'Test.log')
        super().__init__(log_path)
        
        self.record_config()

    def record_config(self):
        with open(os.path.join(self.log_dir,'TestConfig.json'), 'w') as f:
            f.write(json.dumps(self.args))

    def get_model_path(self):
        if hasattr(self, 'model_path'):
            return self.model_path
        else:
            return None

    def get_result_dir(self):
        if hasattr(self, 'result_dir'):
            return self.result_dir
        else:
            return None

if __name__ == "__main__":
    args = Config(train = False).get_config()
    logger = TestLogger(args)
    logger.record_config()
    model_path = logger.get_model_path()
    print(model_path)