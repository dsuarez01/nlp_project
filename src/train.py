from transformers import Trainer, TrainingArguments

class TrainerWrapper:
    def __init__(self, model, training_args):
        self.model = model,
        self.training_args = TrainingArguments(**training_args)
        self.trainer = None

    def train(self, train_dataset, eval_dataset):
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.trainer.train()