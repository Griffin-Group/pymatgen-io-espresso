"""Card classes for the PWin file format"""

from enum import Enum
import logging
from abc import ABC, abstractmethod

import numpy as np


def get_card_header(name, option):
    """
    Prints the card header from the name and options.
    args:
        name: name of the card (str)
        option: option for the card (CardOptions)

    returns:
        card_str: string of the card header

    Example:
        >>> get_card_header("k_points", KPoints.opts.automatic)
        "K_POINTS {automatic}"
    """


class CardOptions(Enum):
    """Enum type of all supported modes for Kpoint generation."""

    def __str__(self):
        return str(self.name)

    def from_string(self, s: str):
        """
        :param s: String
        :return: SupportedOptions
        """
        c = s.lower()[0]
        for m in self:
            if m.name.lower()[0] == c:
                return m
        raise ValueError(f"Can't interpret option {s}")


class PWinCard(ABC):
    def __init__(self, option):
        """Initializes a card's options"""
        if option is None:
            if self.default_deprecated:
                logging.warning(
                    f"No option specified for {self.name} card. This is deprecated, but {self.default_option} will be used by default."
                )
            self.option = self.default_option
        else:
            self.option = self.opts.from_string(option)

    def to_str(self, indent):
        """Initializes a card's header when converting to string"""
        indent = " " * indent
        card_str = f"{self.name.upper()}"
        if self.option:
            card_str += f" {{{self.option}}}"
        return indent, card_str

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def required(self):
        pass

    @property
    @abstractmethod
    def opts(self):
        pass

    @property
    @abstractmethod
    def default_option(self):
        pass

    @property
    @abstractmethod
    def default_deprecated(self):
        pass

    def to_str(self):
        pass


class CellParameters(PWinCard):
    """CELL_PARAMETERS card"""

    class CellParametersOptions(CardOptions):
        alat = 0
        bohr = 1
        angstrom = 2

    name = "cell_parameters"
    required = False
    opts = CellParametersOptions
    default_option = opts.alat
    default_deprecated = True

    def __init__(self, option, data):
        super().__init__(option)
        self.a1 = np.array(data[0])
        self.a2 = np.array(data[1])
        self.a3 = np.array(data[2])

    def to_str(self, indent=2):
        card_str, indent = super().to_str(indent)
        card_str += (
            f"\n{indent}{self.a1[0]:>13.10f}" f" {self.a1[1]:>13.10f}" f" {self.a1[2]:>13.10f}"
        )
        card_str += (
            f"\n{indent}{self.a2[0]:>13.10f}" f" {self.a2[1]:>13.10f}" f" {self.a2[2]:>13.10f}"
        )
        card_str += (
            f"\n{indent}{self.a3[0]:>13.10f}" f" {self.a3[1]:>13.10f}" f" {self.a3[2]:>13.10f}"
        )
        return card_str + "\n"


class AtomicSpecies(PWinCard):
    """ATOMIC_SPECIES card"""

    name = "atomic_species"
    required = True
    opts = None
    default_option = None
    default_deperecated = False

    def __init__(self, data):
        super().__init__(None)
        self.symbols = []
        self.masses = []
        self.files = []
        for item in data:
            self.symbols.append(item[0])
            self.masses.append(item[1])
            self.files.append(item[2])

    def to_str(self, indent=2):
        """Convert card to string"""
        card_str, indent = super().to_str(indent)
        card_str = "".join(
            f"\n{indent}{symbol:>3} {self.masses[i]:>10.6f} {self.files[i]}"
            for i, symbol in enumerate(self.symbols)
        )
        return card_str + "\n"


class AtomicPositions(PWinCard):
    """ATOMIC_POSITIONS card"""

    class AtomicPositionsOptions(CardOptions):
        alat = 0
        bohr = 1
        angstrom = 2
        crystal = 3
        crystal_sg = 4

    name = "atomic_positions"
    required = True
    opts = AtomicPositionsOptions
    default_option = opts.alat
    default_deprecated = True

    def __init__(self, option, data):
        super().__init__(option)
        for item in data:
            self.symbols.append(item[0])
            self.positions.append(item[1:])

    def to_str(self, indent=2):
        card_str, indent = super().to_str(indent)
        card_str = "".join(
            f"\n{indent}{symbol:>3} {self.positions[i][0]:>13.10f} {self.positions[i][1]:>13.10f} {self.positions[i][2]:>13.10f}"
            for i, symbol in enumerate(self.symbols)
        )
        return card_str + "\n"


class KPoints(PWinCard):
    """K_POINTS card"""

    class KPointsOptions(CardOptions):
        automatic = 0
        gamma = 1
        tpiba = 2
        crystal = 3
        tpiba_b = 4
        crystal_b = 5
        tpiba_c = 6
        crystal_c = 7

    name = "k_points"
    required = True
    opts = KPointsOptions
    default_option = opts.tpiba

    def __init__(self, option, data):
        super().__init__(option)
        self.grid = []
        self.shift = []
        self.k = []
        self.weights = []
        self.labels = []
        if self.option == self.opts.automatic:
            k = data[0]
            self.grid = k[:3]
            self.shift = [bool(s) for s in k[3:]]
        elif self.option != self.opts.gamma:
            # Skip first item (number of k-points)
            for k in data[1:]:
                # if len > 4 then we have a label
                label = " ".join(k[4:]).strip("!").lstrip() if len(k) > 4 else ""
                self.k.append(k[:3])
                self.weight.append(k[3])
                self.labels.append(label)

    def to_str(self, indent=2):
        """Convert card to string"""
        card_str, indent = super().to_str(indent)
        if self.option == self.opts.automatic:
            card_str += (
                f"\n{indent}{self.grid[0]:>3}"
                f" {self.grid[1]:>3} {self.grid[2]:>3}"
                f" {int(self.shift[0]):>3}"
                f" {int(self.shift[1]):>3}"
                f" {int(self.shfit[2]):>3}"
            )
        elif self.option != self.opts.gamma:
            card_str += f"\n{len(self.k)}"
            for k, w, l in zip(self.k, self.weights, self.labels):
                card_str += f"\n{indent}{k[0]:>13.10f} {k[1]:>13.10f} {k[2]:>13.10f}"
                card_str += f" {w:>4}" if w == int(w) else f" {w:>10.6f}"
                card_str += f" ! {l}" if l else ""
        return card_str + "\n"
