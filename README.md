# A minimalistic Chatbot

### Requirements Installation
```shell
pip install -r .\requirements.txt
``` 

### Commands
```shell
# Train a bot
python run.py train-chatbot --train_data_path <path/to/train_data.json> --bot_path <path/to/saved_bot>

# Run a bot
python run.py run-chatbot --bot_path <path/to/saved_bot>

# Evaluate a bot
python run.py evaluate-chatbot --bot_path <path/to/saved_bot> --test_data_path <path/to/test_data.json>
```
