from typing import Collection

from tqdm import tqdm

from chatbot.chatbot.chatbot import Chatbot
from chatbot.chatbot.intent import Intent


def evaluate(chatbot: Chatbot, test_data: Collection[Intent]):
    correct = 0
    false = 0
    for intent in tqdm(test_data, desc="Evaluation"):
        for utterance in intent.utterances:
            intent_id, _confidence, _entities = chatbot.predict(utterance)
            if intent.name == chatbot.intents[intent_id].name:
                correct += 1
            else:
                false += 1

    print(correct / (correct + false))
