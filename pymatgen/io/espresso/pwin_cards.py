"""Card classes for the PWin file format"""

from enum import Enum
import logging
from abc import ABC, abstractmethod

import numpy as np


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


class AtomicSpecies(PWinCard):
    """ATOMIC_SPECIES card"""

    name = "atomic_species"
    required = True
    opts = None
    default_option = None
    default_deprecated = False

    def __init__(self, option, data):
        super().__init__(option)
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

    def __init__(self, option, data):
        super().__init__(option)
        self.symbols = [x[0] for x in data]
        self.positions = [np.array(x[1:]) for x in data]

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

    def __init__(self, option, data):
        super().__init__(option)
        self.grid, self.shift, self.k, self.weights, self.labels = [], [], [], [], []
        if self.option == self.opts.automatic:
            self.grid, self.shift = data[0][:3], [bool(s) for s in data[0][3:]]
        elif self.option != self.opts.gamma:
            for k in data[1:]:
                self.k.append(k[:3])
                self.weights.append(k[3])
                self.labels.append(" ".join(k[4:]).strip("!").lstrip() if len(k) > 4 else "")

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


class AdditionalKPoints(PWinCard):
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

    def __init__(self, option, data):
        super().__init__(option)
        self.k, self.weights, self.labels = [], [], []
        for k in data[1:]:
            self.k.append(k[:3])
            self.weights.append(k[3])
            self.labels.append(" ".join(k[4:]).strip("!").lstrip() if len(k) > 4 else "")

    def to_str(self, indent=2):
        """Convert card to string"""
        card_str, indent = super().to_str(indent)
        card_str += f"\n{len(self.k)}"
        for k, w, l in zip(self.k, self.weights, self.labels):
            card_str += f"\n{indent}{k[0]:>13.10f} {k[1]:>13.10f} {k[2]:>13.10f}"
            card_str += f" {w:>4}" if w == int(w) else f" {w:>10.6f}"
            card_str += f" ! {l}" if l else ""
        return card_str + "\n"


class CellParameters(PWinCard):
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

    def __init__(self, option, data):
        super().__init__(option)
        self.a1, self.a2, self.a3 = map(np.array, data)

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


class Constraints(PWinCard):
    """CONSTRAINTS card (not fully implemented)"""

    name = "constraints"
    required = False
    opts = None
    default_option = None
    default_deprecated = False

    def __init__(self, option, data):
        super().__init__(option)
        self.data = data

    def to_str(self, indent=2):
        if not self.data:
            return ""
        card_str, indent = super().to_str(indent)
        card_str += f"\n{self.data}"
        return card_str + "\n"


class Occupations(PWinCard):
    """OCCUPATIONS card (not fully implemented)"""

    name = "occupations"
    required = False
    opts = None
    default_option = None
    default_deprecated = False

    def __init__(self, option, data):
        super().__init__(option)
        self.data = data

    def to_str(self, indent=2):
        if not self.data:
            return ""
        card_str, indent = super().to_str(indent)
        card_str += f"\n{self.data}"
        return card_str + "\n"


class AtomicVelocities(PWinCard):
    """ATOMIC_VELOCITIES card (not fully implemented)"""

    class AtomicVelocitiesOptions(CardOptions):
        au = "a.u."

    name = "atomic_velocities"
    required = False
    opts = AtomicVelocitiesOptions
    # TODO: this card *requires* an option, it has no default
    default_option = opts.au
    default_deprecated = True

    def __init__(self, option, data):
        super().__init__(option)
        self.data = data

    def to_str(self, indent=2):
        if not self.data:
            return ""
        card_str, indent = super().to_str(indent)
        card_str += f"\n{self.data}"
        return card_str + "\n"


class AtomicForces(PWinCard):
    """ATOMIC_FORCES card (not fully implemented)"""

    name = "atomic_forces"
    required = False
    opts = None
    default_option = None
    default_deprecated = False

    def __init__(self, option, data):
        super().__init__(option)
        self.data = data

    def to_str(self, indent=2):
        if not self.data:
            return ""
        card_str, indent = super().to_str(indent)
        card_str += f"\n{self.data}"
        return card_str + "\n"


class Solvents(PWinCard):
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

    def __init__(self, option, data):
        super().__init__(option)
        self.data = data

    def to_str(self, indent=2):
        if not self.data:
            return ""
        card_str, indent = super().to_str(indent)
        card_str += f"\n{self.data}"
        return card_str + "\n"


class Hubbard(PWinCard):
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

    def __init__(self, option, data):
        super().__init__(option)
        self.data = data

    def to_str(self, indent=2):
        if not self.data:
            return ""
        card_str, indent = super().to_str(indent)
        card_str += f"\n{self.data}"
        return card_str + "\n"


class InputCards(Enum):
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
