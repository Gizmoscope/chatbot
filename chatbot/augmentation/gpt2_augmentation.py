import re
from typing import Sequence, Collection

from transformers import pipeline

from chatbot.augmentation.augmentation import Augmentation


class GPT2Augmentation(Augmentation):
    """
    An augmentation class that utilizes a GPT2 model to generate new sentences by using the input sentences as a prompt.

    Supports languages en-UK, en-US and de-DE
    """

    def __init__(self, language):
        if language in ["en-UK", "en-US"]:
            self.pipe = pipeline("gpt2")
            self.prefix = "The following sentence should be rephrased: "
            self.postfix = " For example:"
        elif language == "de-DE":
            self.pipe = pipeline("dbmdz/german-gpt2")
            self.prefix = "Der folgende Satz soll umschrieben werden: "
            self.postfix = " Oder in anderen Worten:"
        else:
            raise ValueError(f"Language {language} not supported!")

    def augment(self, text: Sequence[str], amount: int) -> Collection[str]:
        augmented_texts: list[str] = []
        for i in range(amount):
            text = text[i % len(text)]
            augment_input = self.prefix + text + self.postfix
            augment_output = self.pipe(
                augment_input,
                pad_token_id=50256,
                max_length=len(augment_input.split()) * 5
            )[0]["generated_text"]
            result = augment_output[len(augment_input):]
            result = result.strip()
            result = re.split(r'[\.!\?\n]', result, maxsplit=1)[0] + "."
            augmented_texts.append(result)

        return augmented_texts
