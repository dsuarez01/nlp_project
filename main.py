import pickle
import pandas as pd

import wandb

from src.utils import load_env_file
from src.train import TrainerWrapper
from src.models import BertModelWrapper
from src.data import DataPreprocessor, SongDataset, train_val_split

def main():
    # load env variables
    load_env_file("./.env")

    # initialize wandb
    wandb.init(project='nlp_project')

    # get data
    songs_df = pd.read_csv('./data/year_limited_songs.csv', engine='python')
    
    # instantiate data preprocessor
    encoder_config = {
        'padding':'max_length',
        'truncation':True,
        'max_length':256,
        'return_tensors':'pt',
    }
    preprocessor = DataPreprocessor('./bert-base-uncased', encoder_config)
    encoded_lyrics = preprocessor.encode_lyrics(list(songs_df['lyrics']))
    encoded_tags = preprocessor.encode_tags(songs_df['tag'])
    
    # instantiate song dataset, get train/val split
    song_dataset = SongDataset(
        encoded_lyrics['input_ids'],
        encoded_lyrics['attention_mask'],
        encoded_tags,
    )
    train_dataset, val_dataset = train_val_split(train_size=0.8, dataset=song_dataset)

    # instantiate lora_config + model
    lora_config = {
        'r':16,
        'lora_alpha':16,
        'lora_dropout':0.05,
        'bias':'none',
        'task_type':'SEQ_CLS',
    }
    model_wrapper = BertModelWrapper(
        base_model_name='./bert-base-uncased',
        num_labels=preprocessor.get_num_classes(), # only ever call after encode_tags
        lora_config=lora_config,
    )
    peft_model = model_wrapper.model

    # set up training args and trainer
    training_args = {
        'output_dir':'./results',
        'num_train_epochs':100,
        'per_device_train_batch_size':16,
        'per_device_eval_batch_size':16,
        'warmup_steps':500,
        'weight_decay':0.01,
        'logging_dir':'./logs',
        'logging_steps':10,
        'evaluation_strategy':'epoch',
        'save_strategy':'epoch',
        'dataloader_num_workers':4,
        'dataloader_pin_memory': True,
        'load_best_model_at_end':True,
        'report_to':'wandb',
    }
    
    trainer = TrainerWrapper(peft_model, training_args)
    trainer.train(train_dataset, val_dataset)

    # saving model fine-tuned w/ PEFT, tokenizer and label encoder 
    # to dir ./peft_song_bert_model
    peft_model.save_pretrained('./peft_song_bert_model') 
    tokenizer.save_pretrained('./peft_song_bert_model')

    with open('./peft_song_bert_model/label_encoder.pkl', 'wb') as le_file:
        pickle.dump(label_encoder, le_file)

if __name__ == '__main__':
    main()
