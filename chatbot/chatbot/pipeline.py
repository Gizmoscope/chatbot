from transformers import TokenClassificationPipeline
from transformers.pipelines import AggregationStrategy

from chatbot.constants import SOURCE_LAYER


class ChatbotPipeline(TokenClassificationPipeline):
    """
    A version of the Token Classification Pipeline that does not only return sets of entities but also the relevant
    hidden layer of the transformer model after inference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task="token-classification", aggregation_strategy="simple", **kwargs)

    def _forward(self, model_inputs):
        # Almost identical to super method
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        if self.framework == "tf":
            model_output = self.model(model_inputs.data)
        else:
            model_output = self.model(**model_inputs)

        return {
            "logits": model_output.logits,
            "last_hidden_state": model_output.hidden_states[SOURCE_LAYER],
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            **model_inputs,
        }

    def postprocess(self, model_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=None):
        """
        Apply the post-processing of the super class and add the last hidden layer to the outputs.
        :param model_outputs:
        :param aggregation_strategy:
        :param ignore_labels:
        :return: entities, last hidden layer
        """
        entities = super().postprocess(model_outputs, aggregation_strategy, ignore_labels)
        last_hidden_state = model_outputs["last_hidden_state"].numpy()
        return entities, last_hidden_state, model_outputs["attention_mask"].numpy()
