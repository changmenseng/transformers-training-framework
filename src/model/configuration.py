from transformers import GPT2Config

class GPT2ClassifierConfig(GPT2Config):
    model_type = "gpt2_classifier"
    
    def __init__(
        self,
        n_classes=3,
        sentence_pooling_method='mean',
        **kwargs
    ):
        self.n_classes = n_classes
        super().__init__(**kwargs)
