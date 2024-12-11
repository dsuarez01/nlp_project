from transformers import Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class TrainerWrapper:
    def __init__(self, model, training_args):
        self.model = model
        self.training_args = TrainingArguments(**training_args)
        self.trainer = None
    
    def compute_metrics(self, pred:EvalPrediction):
        # predictions and labels
        predictions = pred.predictions.argmax(axis=-1)
        labels = pred.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
    
    def train(self, train_dataset, eval_dataset, tokenizer):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        self.trainer.train()
