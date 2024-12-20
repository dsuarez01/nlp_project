import torch
from torch.utils.data import Subset, Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, tokenizer_name, encoder_config):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoder_config = encoder_config
        self.label_encoder = LabelEncoder()

    def encode_lyrics(self, lyrics):
        '''
        Expects lyrics to be list.
        '''
        return self.tokenizer(lyrics, **self.encoder_config)

    def encode_tags(self, tags):
        return self.label_encoder.fit_transform(tags)

    def get_num_classes(self):
        '''
        Note: should only call after running encode_tags
        '''
        return len(self.label_encoder.classes_)

class SongDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
def train_val_split(train_size, dataset):
    '''
    Get dataset indices for specified train/val split.
    
    train_size [Float]: fraction of dataset length for training.
    dataset [SongDataset]: a valid instance of SongDataset.
    '''

    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        train_size=train_size,
    )

    return Subset(dataset, train_indices), Subset(dataset, val_indices)