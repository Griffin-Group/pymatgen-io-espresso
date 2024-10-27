"""
This module defines the base input file classes.
"""

import logging
import os
import pathlib
import re
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Any

import f90nml
from monty.json import MSONable

from pymatgen.io.espresso.utils import parse_pwvals


class CardOptions(Enum):
    """Enum type of all supported options for a PWin card."""

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if isinstance(value, str):
            return self.value.lower() == value.lower()
        return self.value.lower() == value.value.lower()

    @classmethod
    def from_string(cls, s: str):
        """
        :param s: String
        :return: SupportedOptions
        """
        for m in cls:
            if m.value.lower() == s.lower():
                return m
        raise ValueError(f"Can't interpret option {s}.")


class BaseInputFile(ABC, MSONable):
    """
    Abstract Base class for input files
    """

    _indent = 2

    def __init__(self, namelists: list[dict[str, Any]], cards: list["InputCard"]):
        namelist_names = [nml.value.name for nml in self.namelist_classes]
        self.namelists = OrderedDict(
            {name: namelists.get(name, None) for name in namelist_names}
        )
        card_names = [c.value.name for c in self.card_classes]
        self.cards = OrderedDict({name: cards.get(name, None) for name in card_names})
        property_names = namelist_names + card_names
        for prop_name in property_names:
            setattr(
                self.__class__,
                prop_name,
                property(
                    self._make_getter(prop_name),
                    self._make_setter(prop_name),
                    self._make_deleter(prop_name),
                ),
            )

    @property
    @abstractmethod
    def namelist_classes(self):
        """All supported namelists as a SupportedInputs enum"""
        pass

    @property
    @abstractmethod
    def card_classes(self):
        """All supported cards as a SupportedCards enum"""
        pass

    def _make_getter(self, name: str):
        """Returns a getter function for a property with name `name`"""
        if name in [n.value.name for n in self.namelist_classes]:
            return lambda self: self.namelists[name]
        elif name in [c.value.name for c in self.card_classes]:
            return lambda self: self.cards[name]

    def _make_setter(self, name: str):
        """Returns a setter function for a property with name `name`"""
        if name in [n.value.name for n in self.namelist_classes]:

            def setter(self, value):
                if value is not None and not isinstance(
                    value, nml := self.namelist_classes.from_string(name)
                ):
                    raise TypeError(f"{name} must be of type {nml}")
                self.namelists[name] = value

            return setter
        elif name in [c.value.name for c in self.card_classes]:

            def setter(self, value):
                if value is not None and not isinstance(
                    value, c := self.card_classes.from_string(name)
                ):
                    raise TypeError(f"{name} must be of type {c}")
                self.cards[name] = value

            return setter

    def _make_deleter(self, name: str):
        """Returns a deleter function for a property with name `name`"""
        if name in [n.value.name for n in self.namelist_classes]:
            return lambda self: self.namelists.__setitem__(name, None)
        elif name in [c.value.name for c in self.card_classes]:
            return lambda self: self.cards.__setitem__(name, None)

    @classmethod
    def from_file(cls, filename: os.PathLike | str) -> "BaseInputFile":
        """
        Reads an inputfile from file

        Args:
            filename: path to file

        Returns:
            PWin object
        """
        parser = f90nml.Parser()
        parser.comment_tokens += "#"

        pwi_str = pathlib.Path(filename).read_text()
        namelists = {}
        for k, v in parser.reads(pwi_str).items():
            Namelist = cls.namelist_classes.from_string(k)
            namelists[k] = Namelist(v)
        cards = cls._parse_cards(pwi_str)

        return cls(namelists, cards)

    @classmethod
    def _parse_cards(cls, pwi_str: str) -> dict[str, "InputCard"]:
        card_strings = pwi_str.rsplit("/", 1)[1].split("\n")
        card_strings = [c for c in card_strings if c]
        card_idx = [
            i
            for i, string in enumerate(card_strings)
            if string.split()[0].lower() in [c.value.name for c in cls.card_classes]
        ]
        cards = {}
        for i, j in zip(card_idx, card_idx[1:] + [None]):  # type: ignore
            card_name = card_strings[i].split()[0].lower()
            card_string = "\n".join(card_strings[i:j])
            Card = cls.card_classes.from_string(card_name)
            cards[card_name] = Card.from_string(card_string)

        return cards

    def validate(self) -> bool:
        """
        Very basic validation for the input file.
        Currently only checks that required namelists and cards are present.
        """
        required_namelists = [
            nml.value.name for nml in self.namelist_classes if nml.value.required
        ]
        if any(self.namelists[nml] is None for nml in required_namelists):
            msg = "Input file is missing required namelists:"
            for nml in required_namelists:
                if self.namelists[nml] is None:
                    msg += f" &{nml.upper()}"
            warnings.warn(msg, EspressoInputWarning)
            return False

        required_cards = [c.value.name for c in self.card_classes if c.value.required]
        if any(self.cards[card] is None for card in required_cards):
            msg = "Input file is missing required cards:"
            for card in required_cards:
                if self.cards[card] is None:
                    msg += f" {card.upper()}"
            warnings.warn(msg, EspressoInputWarning)
            return False

        return True

    def __str__(self):
        """
        Return the input file as a string
        """
        string = ""
        for nml in self.namelists.values():
            if nml is not None:
                nml.indent = self._indent
                string += str(nml) + "\n"

        for c in self.cards.values():
            if c is not None:
                c.indent = self._indent
                string += str(c) + "\n"

        return string

    def to_file(self, filename: os.PathLike | str, indent: int = 2):
        """
        Write the input file to a file.

        Args:
            filename: path to file
            indent: number of spaces to use for indentation
        """
        self._indent = indent
        with open(filename, "wb") as f:
            f.write(self.__str__().encode("ascii"))


class InputNamelist(ABC, OrderedDict):
    """
    Abstract Base class for namelists in input files
    """

    indent = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        """
        Convert namelist to string
        """
        nl = f90nml.Namelist({self.name: self})
        nl.indent = self.indent * " "
        string = str(nl)
        return re.sub(r"^&(\w+)", lambda m: m.group(0).upper(), string)

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def required(self):
        pass


class InputCard(ABC):
    """
    Abstract Base class for cards in input files

    Args:
        option (str): The option for the card (e.g., "RELAX")
        body (list): The body of the card
    """

    indent = 2

    def __init__(self, option: str | CardOptions, body: str):
        """
        Args:
            option (str): The option for the card (e.g., "RELAX")
            body (list): The body of the card
        """
        if isinstance(option, str):
            option = self.opts.from_string(option)
        self.option = option
        self._body = body

    @property
    @abstractmethod
    def name(self, value):
        pass

    @property
    @abstractmethod
    def required(self, value):
        pass

    @property
    @abstractmethod
    def opts(self, value):
        pass

    @property
    @abstractmethod
    def default_option(self, value):
        pass

    @property
    @abstractmethod
    def default_deprecated(self, value):
        pass

    def __str__(self):
        """
        Convert card to string
        This implementation is for generic (i.e., not fully implemented) cards
        """
        return self.get_header() + self.get_body(" " * self.indent)

    # TODO: this should become an abstract method when all cards are implemented
    def get_body(self, indent: str) -> str:
        """
        Convert card body to string
        This implementation is for generic (i.e., not fully implemented) cards
        """
        return "".join(
            f"\n{indent}{' '.join(line) if isinstance(line, list) else line}"
            for line in self._body
        )

    @property
    def body(self):
        return self.get_body(self.indent)

    @classmethod
    def from_string(cls, s: str) -> "InputCard":
        """
        Create card object from string
        This implementation is for generic (i.e., not fully implemented) cards
        """
        option, body = cls.split_card_string(s)
        return cls(option, body)

    @classmethod
    def get_option(cls, option: str) -> CardOptions:
        """Initializes a card's options"""
        if option is not None:
            return cls.opts.from_string(option)
        if cls.default_deprecated:
            logging.warning(
                f"No option specified for {cls.name} card. This is deprecated, but {cls.default_option} will be used by default."
            )
        return cls.default_option

    @classmethod
    def split_card_string(cls, s: str) -> tuple[str, list]:
        """
        Splits a card into an option and a list of values of the correct type.
        :param s: String containing a card (as it would appear in a PWin file)
        :return: option: string for the card's option or None
                 values: list of lists of values for the card

        Example:
        >>> s = "ATOMIC_SPECIES\nH 1.00794 H.UPF\nO 15.9994 O.UPF"
        >>> option, values = InputCard.split_card_string_string(s)
        >>> option, values
        >>> (None, [["H", 1.00794, "H.UPF"], ["O", 15.9994, "O.UPF"]])
        """
        s = re.sub(r"\t", " ", s)
        header = s.strip().split("\n")[0]
        body = s.strip().split("\n")[1:]
        if len(header.split()) > 1:
            option = re.sub(r"[()]", "", header.split()[1])
            option = option.lower()
            option = re.sub(r"[()]", "", option)
            option = re.sub(r"[{}]", "", option)
        else:
            option = None
        return cls.get_option(option), parse_pwvals(body)

    def get_header(self) -> str:
        """Gets a card's header as a string"""
        if self.name is None:
            return ""
        header = f"{self.name.upper()}"
        if self.option:
            header += f" {{{self.option}}}"
        return header


class SupportedInputs(Enum):
    """Enum type of all supported input cards and namelists."""

    @classmethod
    def from_string(cls, s: str):
        """
        :param s: String
        :return: InputCard or InputNamelist
        """
        for m in cls:
            if m.name.lower() == s.lower():
                return m.value
        raise ValueError(f"Can't interpret card or namelist {s}.")


# Custom warning for invalid input
class EspressoInputWarning(UserWarning):
    """
    Warning class for PWin parsing.
    """
