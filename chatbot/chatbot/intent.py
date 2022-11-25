import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain, combinations
from typing import Collection


@dataclass
class Intent:
    """
    An Intent represents a collection of user inputs the chatbot is able to respond to. It consists of a list of example
    utterances that trigger the intent and a list of possible responses.
    The responses might contain placeholders for entities which usually are taken from the user input that triggred the
    given intent.
    """

    name: str
    responses: list[str]
    utterances: list[str]

    def __post_init__(self):
        # Find entity placeholders e.g. [PER] or [LOC]
        regex = re.compile(r"\[([A-Z]*)\]")

        # group the responses according to the used entity placeholders
        self.responses_by_entity_groups = defaultdict(list)
        for response in self.responses:
            entity_groups = [match for match in regex.findall(response)]
            key = get_key(entity_groups)
            self.responses_by_entity_groups[key].append(response)

    def respond_using_entities(self, entities: list[dict[str, str]]) -> str:
        """
        Choose/create a response based on the given set of entities
        :param entities: a list of entities
        :return: a response based on the given entities
        """
        entity_groups = [entity["entity_group"] for entity in entities]

        # determine all subsets of the given intent set. Start with the largest one
        subsets = chain.from_iterable(combinations(entity_groups, r) for r in range(len(entity_groups), -1, -1))

        for subset in subsets:
            key = get_key(subset)
            # the first response that fits the first of these subsets is used as a template
            if key in self.responses_by_entity_groups:
                template = random.choice(self.responses_by_entity_groups[key])

                # replace all placeholders by the entities
                for entity_group in subset:
                    entity = [e["word"] for e in entities if e["entity_group"] == entity_group][0]
                    template = template.replace(f"[{entity_group}]", entity)

                return template

        return "ERROR: Couldn't resolve entities"


def get_key(set_of_entity_groups: Collection[str]) -> str:
    """
    Convert a list or set of strings into a hashable object
    :param set_of_entity_groups: a list of strings
    :return: a string
    """
    return "-".join(sorted(set_of_entity_groups))
