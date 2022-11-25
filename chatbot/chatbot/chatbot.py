from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModelForTokenClassification

from chatbot.augmentation.augmentation import Augmentation
from chatbot.augmentation.dummy_augmentation import DummyAugmentation
from chatbot.chatbot.intent import Intent
from chatbot.chatbot.pipeline import ChatbotPipeline
from chatbot.constants import TXT_ENCODING, CONFIDENCE_THRESHOLD, BATCH_SIZE, EPOCHS, SOURCE_LAYER


class Chatbot:
    """
    A chatbot is a collection of intents, i.e. bundles of utterances the bot knows how to respond to.
    The list of intents must be specified using in the constructor. Technically, the bot could be used right away.
    However, `train()` should be called first because otherwise the bot cannot understand the intents well.
    """

    def __init__(self, intents: list[Intent]):
        self.intents = intents
        self.tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
        self.base_model = TFAutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl", output_hidden_states=True)
        self.pipeline = ChatbotPipeline(model=self.base_model, tokenizer=self.tokenizer)
        self.downstream_model = get_downstream_model(len(intents))
        self.augmentation: Augmentation = DummyAugmentation()

    def train(self):
        """
        Train the chatbot.
        """
        # extract the relevant data from the intents
        utterances = []
        intent_ids = []
        for intent_id, intent in enumerate(tqdm(self.intents, desc="Preprocessing")):
            augmented_utterances = [
                *intent.utterances,  self.augmentation.augment(intent.utterances, len(intent.utterances))
            ]
            utterances.extend(augmented_utterances)
            intent_ids.extend([intent_id] * len(augmented_utterances))

        # use the base model to generate the features for the downstream model
        features = []
        attention_mask = []
        for batch in tqdm(range(0, len(utterances), BATCH_SIZE), desc="Feature Extraction"):
            tokens = self.tokenizer(utterances[batch:batch + BATCH_SIZE], padding=True, return_tensors="tf")
            attention_mask.extend([encoding.attention_mask for encoding in tokens.encodings])
            features.extend(self.base_model(tokens).hidden_states[SOURCE_LAYER].numpy())

        # fit the downstream model
        self.downstream_model.fit(
            x=(_merge_features(features), _merge_attentions(attention_mask)),
            y=one_hot_encoding(intent_ids, max(intent_ids) + 1),
            shuffle=True,
            epochs=EPOCHS
        )

    def save(self, path: Path):
        """
        Save the chatbot.
        :param path: destination
        """
        self.downstream_model.save(path)
        (path / "intents").write_text(json.dumps([asdict(intent) for intent in self.intents]), encoding=TXT_ENCODING)

    @classmethod
    def load(cls, path: Path) -> Chatbot:
        """
        Loaf the chatbot
        :param path: source
        :return: loaded chatbot
        """
        intents = json.loads((path / "intents").read_text(encoding=TXT_ENCODING))
        chatbot = Chatbot([Intent(**intent) for intent in intents])
        chatbot.downstream_model = tf.keras.models.load_model(path)
        return chatbot

    def predict(self, utterance: str) -> tuple[int, float, list[dict[str, str]]]:
        """
        Predict intent and entities for the given utterance.
        :param utterance: Any text
        :return: (intent_id, confidence, [entity1, entity2, ...])
        """
        entities, last_hidden_state, attention_mask = self.pipeline(utterance)
        intent_distribution = self.downstream_model((last_hidden_state, attention_mask)).numpy()
        intent_id = np.argmax(intent_distribution)
        confidence = intent_distribution[0, intent_id]
        return int(intent_id), float(confidence), entities

    def respond(self, utterance: str) -> str:
        """
        Respond to the given utterance. In order to do so, the method determines the intent behind the utterance and the
        entities used within the utterances.
        :param utterance: Any text
        :return: A response based on the given utterance
        """
        intent_id, confidence, entities = self.predict(utterance)

        if confidence > CONFIDENCE_THRESHOLD:
            return self.intents[intent_id].respond_using_entities(entities)
        else:
            return "Ich habe dich leider nicht verstanden. Kannst du das bitte anders vormulieren?"


def get_downstream_model(num_classes: int) -> tf.keras.Model:
    """
    Build a downstream model.
    The model takes a sequence of embeddings (dim=768) and an attention mask and produces a probability distribution.
    :param num_classes: number of output classes for the distribution
    :return: a keras model
    """
    input_layer = tf.keras.layers.Input(shape=(None, 768), name="input")
    attention_mask = tf.keras.layers.Input(shape=(None, 1), name="attention_mask")

    # First Attention Layer
    x = tf.keras.layers.Dropout(0.1)(input_layer)
    x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)(x, x, attention_mask=attention_mask) + x

    # Second Attention Layer
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)(x, x, attention_mask=attention_mask) + x

    # Dense Layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x[:, 0])

    transformer_model = tf.keras.Model(inputs=[input_layer, attention_mask], outputs=[output])

    transformer_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        metrics=["accuracy"]
    )

    return transformer_model


def one_hot_encoding(ids: list[int], num_classes: int) -> np.ndarray:
    """
    :param ids: list of integers
    :param num_classes: number of classes for the one hot encoding
    :return: one hot encoding representation of the ids
    """
    return np.array([[float(id == label) for label in range(num_classes)] for id in ids])


def _merge_features(arrays: list[np.ndarray]) -> np.ndarray:
    longest = max([array.shape[0] for array in arrays])
    return np.stack([np.pad(array, ((0, longest - array.shape[0]), (0, 0))) for array in arrays])


def _merge_attentions(arrays: list[list[int]]) -> np.ndarray:
    longest = max([len(array) for array in arrays])
    return np.stack([np.pad(array, (0, longest - len(array))) for array in arrays])
