from transformers import BertForSequenceClassification
from transformers import DistilBertForSequenceClassification

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict, # currently not in use
    prepare_model_for_kbit_training, # currently not in use
    set_peft_model_state_dict, # currently not in use
)

class BertModelWrapper:
    def __init__(self, base_model_name, num_labels, lora_config):
        super().__init__()
        self.base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
        self.lora_config = LoraConfig(**lora_config)
        self.model = get_peft_model(self.base_model, self.lora_config)

class DistilBertModelWrapper:
    def __init__(self, base_model_name, num_labels, lora_config):
        super().__init__()
        self.base_model = DistilBertForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
        self.lora_config = LoraConfig(**lora_config)
        self.model = get_peft_model(self.base_model, self.lora_config)