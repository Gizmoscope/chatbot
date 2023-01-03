import json
from pathlib import Path

import click

from chatbot.constants import TXT_ENCODING


@click.group()
def cli():
    pass


@cli.command()
@click.option("--train_data_path", type=Path, required=True)
@click.option("--bot_path", type=Path, required=True)
def train_chatbot(train_data_path: Path, bot_path: Path):
    from chatbot.chatbot.chatbot import Chatbot

    chatbot = Chatbot.define(train_data_path)
    chatbot.train()
    chatbot.save(bot_path)


@cli.command()
@click.option("--bot_path", type=Path, required=True)
def run_chatbot(bot_path: Path):
    from chatbot.chatbot.chatbot import Chatbot

    chatbot = Chatbot.load(bot_path)
    print(chatbot.greeting)
    while True:
        utterance = input("> ")
        print(chatbot.respond(utterance))


@cli.command()
@click.option("--bot_path", type=Path, required=True)
@click.option("--test_data_path", type=Path, required=True)
def evaluate_chatbot(bot_path: Path, test_data_path: Path):
    from chatbot.chatbot.chatbot import Chatbot
    from chatbot.chatbot.intent import Intent
    from chatbot.evaluation.evaluation import evaluate

    chatbot = Chatbot.load(bot_path)
    test_data = json.loads(test_data_path.read_text(encoding=TXT_ENCODING))
    evaluate(chatbot, [Intent(**intent) for intent in test_data["intents"]])
