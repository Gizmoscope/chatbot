from abc import ABC, abstractmethod
from typing import Sequence


class Augmentation(ABC):
    """
    Base class for all augmentations.
    """

    @abstractmethod
    def augment(self, texts: Sequence[str], amount: int) -> Sequence[str]:
        """
        :param texts: list of texts
        :param amount: number of texts as output
        :return: list of texts from request_texts with amount as length
        """
