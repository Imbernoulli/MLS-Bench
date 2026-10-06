import json

class Config(object):
    def __init__(self, config, train=True):
        if train:
            self.mode = 'train'
        else:
            self.mode = 'test'

        if self.mode == 'train':        
            with open(f'config/{config}.json', 'r') as f:
                self.train_config = json.load(f)['train']
        elif self.mode == 'test':
            with open(f'config/{config}.json', 'r') as f:
                self.test_config = json.load(f)['test']

    def get_mode(self):
        return self.mode

    def get_config(self):
        if self.mode == 'train':
            return self.train_config
        elif self.mode == 'test':
            return self.test_config

    def show_config(self, train=True):
        print('='*50)
        if self.mode == 'train':
            for key, value in self.train_config.items():
                print(f'{key}: {value}')
        elif self.mode == 'test': 
            for key, value in self.test_config.items():
                print(f'{key}: {value}')
        print('='*50)

if __name__ == '__main__':
    # demo
    config = Config()
    args = config.get_config()
