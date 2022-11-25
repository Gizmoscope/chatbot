from typing import Collection

from sklearn.metrics import classification_report
from tqdm import tqdm

from chatbot.chatbot.chatbot import Chatbot
from chatbot.chatbot.intent import Intent


def evaluate(chatbot: Chatbot, test_data: Collection[Intent]):
    """
    Evaluate the given chatbot.
    :param chatbot: a chatbot
    :param test_data: data used to evaluate the bot (should be different from train data)
    """
    y_true = []
    y_pred = []
    for intent in tqdm(test_data, desc="Evaluation"):
        for utterance in intent.utterances:
            intent_id, _confidence, _entities = chatbot.predict(utterance)
            y_true.append(intent.name)
            y_pred.append(chatbot.intents[intent_id].name)

    print(classification_report(y_true, y_pred))
