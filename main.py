import pickle
import pandas as pd

import wandb

from src.utils import load_env_file
from src.train import TrainerWrapper
from src.models import BertModelWrapper, DistilBertModelWrapper
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
        'max_length':512,
        'return_tensors':'pt',
    }
    preprocessor = DataPreprocessor('./distilbert-base-uncased', encoder_config)
    encoded_lyrics = preprocessor.encode_lyrics(list(songs_df['lyrics']))
    encoded_tags = preprocessor.encode_tags(songs_df['tag'])
    
    # save the tokenizer, pickle the label encoder
    preprocessor.tokenizer.save_pretrained('./peft_song_bert_model/dbert_run_100')
    with open('./peft_song_bert_model/dbert_run_100/label_encoder.pkl', 'wb') as le_file:
        pickle.dump(preprocessor.label_encoder, le_file)

    # instantiate song dataset, get train/val split
    song_dataset = SongDataset(
        encoded_lyrics['input_ids'],
        encoded_lyrics['attention_mask'],
        encoded_tags,
    )
    train_dataset, val_dataset = train_val_split(train_size=0.8, dataset=song_dataset)

    # instantiate lora_config + model
    lora_config = {
        'r': 32,  # higher rank = higher expressivity
        'lora_alpha': 64,  # higher alpha = better scaling
        'lora_dropout': 0.1,  # increase dropout = prevent overfitting
        'bias': 'none',
        'target_modules': 'all-linear',
        'task_type': 'SEQ_CLS',
    }

    model_wrapper = DistilBertModelWrapper(
        base_model_name='./distilbert-base-uncased',
        num_labels=preprocessor.get_num_classes(), # only ever call after encode_tags
        lora_config=lora_config,
    )
    peft_model = model_wrapper.model

    # set up training args and trainer
    training_args = {
        'output_dir': './results/dbert_run_100',
        'run_name': 'dbert_run_100',
        'num_train_epochs': 100,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'gradient_accumulation_steps': 2,
        'learning_rate': 3e-5,
        'fp16': True,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'logging_dir': './logs',
        'logging_steps': 50,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'save_total_limit': 3,
        'dataloader_num_workers': 0,
        'dataloader_pin_memory': True,
        'load_best_model_at_end': True,
        'report_to': 'wandb',
        'disable_tqdm': True,
    }
    
    trainer = TrainerWrapper(peft_model, training_args)
    trainer.train(train_dataset, val_dataset, preprocessor.tokenizer)

    # saving model fine-tuned w/ PEFT
    peft_model.save_pretrained('./peft_song_bert_model/dbert_run_100')

if __name__ == '__main__':
    main()
