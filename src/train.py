from transformers import Trainer, TrainingArguments


class TrainerWrapper:
    def __init__(self, model, training_args):
        self.model = model
        self.training_args = TrainingArguments(**training_args)
        self.trainer = None

    def compute_metrics(self, pred):
        pred = pred.predictions.argmax(-1)
        labels = pred.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='weighted')
        acc = accuracy_score(labels, pred)
        return {'accuracy': acc, "f1_score": f1}

    def train(self, train_dataset, eval_dataset, tokenizer):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        self.trainer.train()
