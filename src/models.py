from transformers import BertForSequenceClassification

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict, # not in use
    prepare_model_for_kbit_training, # not in use
    set_peft_model_state_dict, # not in use
)

class BertModel:
    def __init__(self, base_model_name, num_labels, lora_config):
        super().__init__()
        self.base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
        self.lora_config = LoraConfig(**lora_config)
        self.model = get_peft_model(self.base_model, self.lora_config)