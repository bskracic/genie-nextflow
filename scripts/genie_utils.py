import json


def load_config(file):
    with open(file, 'r') as fh:
        config = json.load(fh)

    return config
