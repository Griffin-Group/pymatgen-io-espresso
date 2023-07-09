"""Card classes for the PWin file format"""

from enum import Enum
import logging
from abc import ABC, abstractmethod
import re

import numpy as np
from pymatgen.io.espresso.utils import parse_pwvals


class CardOptions(Enum):
    """Enum type of all supported modes for a PWin card."""

    def __str__(self):
        return str(self.value)

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


class InputCard(ABC):
    indent = 2

    def __init__(self, option, body):
        self.option = option
        self.body = body

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
        header, indent = self.get_header()
        card_str = "".join(
            f"\n{indent}{' '.join(line) if isinstance(line, list) else line}" for line in self.body
        )
        return header + card_str + "\n"

    @classmethod
    def from_string(cls, s: str):
        """
        Create card object from string
        This implementation is for generic (i.e., not fully implemented) cards
        """
        option, body = cls.split_card_string(s)
        return cls(option, body)

    @classmethod
    def get_option(cls, option):
        """Initializes a card's options"""
        if option is not None:
            return cls.opts.from_string(option)
        if cls.default_deprecated:
            logging.warning(
                f"No option specified for {cls.name} card. This is deprecated, but {cls.default_option} will be used by default."
            )
        return cls.default_option

    @classmethod
    def split_card_string(cls, s: str):
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

    def get_header(self):
        """Initializes a card's header when converting to string"""
        indent = " " * self.indent
        header = f"{self.name.upper()}"
        if self.option:
            header += f" {{{self.option}}}"
        return header, indent


class AtomicSpecies(InputCard):
    """ATOMIC_SPECIES card"""

    name = "atomic_species"
    required = True
    opts = None
    default_option = None
    default_deprecated = False

    def __init__(self, option, symbols, masses, files):
        self.option = option
        self.symbols = symbols
        self.masses = masses
        self.files = files

    def __str__(self):
        """Convert card to string"""
        header, indent = super().get_header()
        card_str = "".join(
            f"\n{indent}{symbol:>3} {self.masses[i]:>10.6f} {self.files[i]}"
            for i, symbol in enumerate(self.symbols)
        )
        return header + card_str + "\n"

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        symbols = [item[0] for item in body]
        masses = [item[1] for item in body]
        files = [item[2] for item in body]
        return cls(option, symbols, masses, files)


class AtomicPositions(InputCard):
    """ATOMIC_POSITIONS card"""

    class AtomicPositionsOptions(CardOptions):
        alat = "alat"
        bohr = "bohr"
        angstrom = "angstrom"
        crystal = "crystal"
        crystal_sg = "crystal_sg"

    name = "atomic_positions"
    required = True
    opts = AtomicPositionsOptions
    default_option = opts.alat
    default_deprecated = True

    def __init__(self, option, symbols, positions):
        self.option = option
        self.symbols = symbols
        self.positions = positions

    def __str__(self):
        header, indent = super().get_header()
        card_str = "".join(
            f"\n{indent}{symbol:>3} {self.positions[i][0]:>13.10f} {self.positions[i][1]:>13.10f} {self.positions[i][2]:>13.10f}"
            for i, symbol in enumerate(self.symbols)
        )
        return header + card_str + "\n"

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        symbols = [line[0] for line in body]
        positions = [np.array(line[1:]) for line in body]
        return cls(option, symbols, positions)


class KPoints(InputCard):
    """K_POINTS card"""

    class KPointsOptions(CardOptions):
        automatic = "automatic"
        gamma = "gamma"
        tpiba = "tpiba"
        crystal = "crystal"
        tpiba_b = "tpiba_b"
        crystal_b = "crystal_b"
        tpiba_c = "tpiba_c"
        crystal_c = "crystal_c"

    name = "k_points"
    required = True
    opts = KPointsOptions
    default_option = opts.tpiba
    default_deprecated = False

    def __init__(self, option, grid, shift, k, weights, labels):
        self.option = option
        self.grid = grid
        self.shift = shift
        self.k = k
        self.weights = weights
        self.labels = labels

    def __str__(self):
        """Convert card to string"""
        header, indent = super().get_header()
        if self.option == self.opts.automatic:
            card_str = (
                f"\n{indent}{self.grid[0]:>3}"
                f" {self.grid[1]:>3} {self.grid[2]:>3}"
                f" {int(self.shift[0]):>3}"
                f" {int(self.shift[1]):>3}"
                f" {int(self.shift[2]):>3}"
            )
        elif self.option != self.opts.gamma:
            card_str = f"\n{len(self.k)}"
            for k, w, l in zip(self.k, self.weights, self.labels):
                card_str += f"\n{indent}{k[0]:>13.10f} {k[1]:>13.10f} {k[2]:>13.10f}"
                card_str += f" {w:>4}" if w == int(w) else f" {w:>10.6f}"
                card_str += f" ! {l}" if l else ""
        return header + card_str + "\n"

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        grid, shift, k, weights, labels = [], [], [], [], []
        if option == cls.opts.automatic:
            grid, shift = body[0][:3], [bool(s) for s in body[0][3:]]
        elif option != cls.opts.gamma:
            for line in body[1:]:
                k.append(line[:3])
                weights.append(line[3])
                labels.append(" ".join(line[4:]).strip("!").lstrip() if len(line) > 4 else "")

        return cls(option, grid, shift, k, weights, labels)


class AdditionalKPoints(InputCard):
    """ADDITIONAL_K_POINTS card"""

    class AdditionalKPointsOptions(CardOptions):
        tpiba = "tpiba"
        crystal = "crystal"
        tpiba_b = "tpiba_b"
        crystal_b = "crystal_b"
        tpiba_c = "tpiba_c"
        crystal_c = "crystal_c"

    name = "additional_k_points"
    required = False
    opts = AdditionalKPointsOptions
    default_option = opts.tpiba
    default_deprecated = False

    def __init__(self, option, k, weights, labels):
        self.option = option
        self.k = k
        self.weights = weights
        self.labels = labels

    def __str__(self):
        """Convert card to string"""
        header, indent = super().get_header()
        card_str = f"\n{len(self.k)}"
        for k, w, l in zip(self.k, self.weights, self.labels):
            card_str += f"\n{indent}{k[0]:>13.10f} {k[1]:>13.10f} {k[2]:>13.10f}"
            card_str += f" {w:>4}" if w == int(w) else f" {w:>10.6f}"
            card_str += f" ! {l}" if l else ""
        return header + card_str + "\n"

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        k, weights, labels = [], [], []
        for line in body[1:]:
            k.append(line[:3])
            weights.append(line[3])
            labels.append(" ".join(line[4:]).strip("!").lstrip() if len(line) > 4 else "")

        return cls(option, k, weights, labels)


class CellParameters(InputCard):
    """CELL_PARAMETERS card"""

    class CellParametersOptions(CardOptions):
        alat = "alat"
        bohr = "bohr"
        angstrom = "angstrom"

    name = "cell_parameters"
    required = False
    opts = CellParametersOptions
    default_option = opts.alat
    default_deprecated = True

    def __init__(self, option, a1, a2, a3):
        self.option = option
        self.a1, self.a2, self.a3 = a1, a2, a3

    def __str__(self):
        header, indent = super().get_header()
        card_str = (
            f"\n{indent}{self.a1[0]:>13.10f}" f" {self.a1[1]:>13.10f}" f" {self.a1[2]:>13.10f}"
        )
        card_str += (
            f"\n{indent}{self.a2[0]:>13.10f}" f" {self.a2[1]:>13.10f}" f" {self.a2[2]:>13.10f}"
        )
        card_str += (
            f"\n{indent}{self.a3[0]:>13.10f}" f" {self.a3[1]:>13.10f}" f" {self.a3[2]:>13.10f}"
        )
        return header + card_str + "\n"

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        a1, a2, a3 = map(np.array, body)
        return cls(option, a1, a2, a3)


class Constraints(InputCard):
    """CONSTRAINTS card (not fully implemented)"""

    name = "constraints"
    required = False
    opts = None
    default_option = None
    default_deprecated = False


class Occupations(InputCard):
    """OCCUPATIONS card (not fully implemented)"""

    name = "occupations"
    required = False
    opts = None
    default_option = None
    default_deprecated = False


class AtomicVelocities(InputCard):
    """ATOMIC_VELOCITIES card (not fully implemented)"""

    class AtomicVelocitiesOptions(CardOptions):
        au = "a.u."

    name = "atomic_velocities"
    required = False
    opts = AtomicVelocitiesOptions
    # TODO: this card *requires* an option, it has no default
    default_option = opts.au
    default_deprecated = True


class AtomicForces(InputCard):
    """ATOMIC_FORCES card (not fully implemented)"""

    name = "atomic_forces"
    required = False
    opts = None
    default_option = None
    default_deprecated = False


class Solvents(InputCard):
    """SOLVENTS card (not fully implemented)"""

    class SolventsOptions(CardOptions):
        cell = "1/cell"
        molL = "mol/L"
        gcm3 = "g/cm^3"

    name = "solvents"
    required = False
    opts = SolventsOptions
    # TODO: this card *requires* an option, it has no default
    default_option = None
    default_deprecated = False


class Hubbard(InputCard):
    """HUBBARD card (not fully implemented)"""

    class HubbardOptions(CardOptions):
        atomic = "atomic"
        othoatomic = "ortho-atomic"
        normatomic = "norm-atomic"
        wf = "wf"
        pseudo = "pseudo"

    name = "hubbard"
    required = False
    opts = HubbardOptions
    # TODO: this card *requires* an option, it has no default
    default_option = opts.atomic
    default_deprecated = True


class PWinCards(Enum):
    """Enum type of all supported input cards."""

    atomic_species = AtomicSpecies
    atomic_positions = AtomicPositions
    k_points = KPoints
    additional_k_points = AdditionalKPoints
    cell_parameters = CellParameters
    constraints = Constraints
    occupations = Occupations
    atomic_velocities = AtomicVelocities
    atomic_forces = AtomicForces
    solvents = Solvents
    hubbard = Hubbard

    @classmethod
    def from_string(cls, s: str):
        """
        :param s: String
        :return: SupportedCards
        """
        for m in cls:
            if m.name.lower() == s.lower():
                return m.value
        raise ValueError(f"Can't interpret card {s}.")
