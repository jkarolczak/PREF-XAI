from typing import TypeAlias, Literal

from lib.experts.cards import cards_deck  # noqa
from lib.experts.diversify import (diversify_by_properties,
                                   diversify_by_spanning_examples)  # noqa
from lib.experts.top import top_3  # noqa

TExpert: TypeAlias = Literal["diversify_by_properties", "diversify_by_spanning_examples", "top_3", "cards_deck"]


def get_expert(name: str) -> callable:
    """Returns the expert function based on the name.

    Args:
        name (str): The name of the expert function.

    Returns:
        callable: The expert function.
    """

    return globals()[name]
