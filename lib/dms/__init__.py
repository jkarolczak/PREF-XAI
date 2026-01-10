from typing import TypeAlias, Literal

# from lib.dms.cards_dm import cards_dm  # noqa
from lib.dms.random_dm import RandomDM  # noqa

TExpert: TypeAlias = Literal["random"]  # "cards_deck"


def get_dm(name: str) -> callable:
    """Returns the expert function based on the name.

    Args:
        name (str): The name of the expert function.

    Returns:
        callable: The expert function.
    """

    return globals()[name]
