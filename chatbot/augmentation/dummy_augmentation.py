from typing import Sequence

from chatbot.augmentation.augmentation import Augmentation


class DummyAugmentation(Augmentation):
    """
    Simple augmentation implementation that only repeats the input sentences.
    """

    def augment(self, texts: Sequence[str], amount: int) -> Sequence[str]:
        augmented_texts: list[str] = []
        if len(texts) == 0:
            return augmented_texts
        for i in range(amount):
            augmented_texts.append(texts[i % len(texts)])
        return augmented_texts
