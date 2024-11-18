import os

def load_env_file(filepath):
    '''
    Load env variables from a key-value formatted .env file.
    '''
    with open(filepath) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            os.environ[key] = value