import os
import torch
import yaml
from pathlib import Path

def load_latest_model_from(location, model_name, useCuda=True):
    files = [location + "/" + f for f in os.listdir(location)]
    files = [f for f in files if model_name in f]
    newest_file = max(files, key=os.path.getctime)
    print('Loading last saved model: ' + newest_file)
    if useCuda:
        model = torch.load(newest_file)
    else:
        model = load_to_cpu(newest_file)
        model = param_keys_to_cpu(model)
    return model

def param_keys_to_cpu(model):
    from collections import OrderedDict
    new_model = OrderedDict()
    for k, v in model.items():
        name = k[7:] # remove `module.`
        new_model[name] = v
    return new_model

def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    return model

def get_model_name(settings):
    modelName = settings['data']['model_name']+'_'+settings['model']['classifier']['type']
    if settings['model']['encoder']['pretraining'] is not None:
        modelName += '_enc_'+settings['model']['encoder']['pretraining']
        modelName += ('_finetune' if settings['model']['encoder']['finetune'] else '_frozen')
    else:
        modelName += '_enc_scratch'

    return modelName
    
class SettingsLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(SettingsLoader, self).__init__(stream)
    
    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, YAMLLoader)

SettingsLoader.add_constructor('!include', SettingsLoader.include)

def load_settings(file_path):
    with file_path.open('r') as f:
        return yaml.load(f, Loader=SettingsLoader)

