"""
Synthetic French business record augmentation.

Generates plausible French-style business records from generic linguistic patterns
(NO real French business registry data).

Purpose: Stress-test pipeline generalization to the unseen France test segment
by injecting synthetic French records into a held-out validation subset.
"""
from __future__ import annotations

import random
import re
from typing import Optional

import polars as pl
from loguru import logger


# Generic French linguistic patterns (public domain, not real records)
_FR_LEGAL_SUFFIXES = [
    "SARL", "SAS", "SA", "EURL", "SASU", "SNC", "SCI",
    "S.A.R.L.", "S.A.S.", "S.A.",
    "Sarl", "Sas", "Eurl",
]

_FR_STREET_PREFIXES = [
    "Rue", "Avenue", "Boulevard", "Impasse", "Allée",
    "Place", "Chemin", "Route", "Passage", "Cour",
]

_FR_CITIES = [
    "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse",
    "Nice", "Nantes", "Strasbourg", "Montpellier", "Rennes",
    "Grenoble", "Lille", "Dijon", "Angers", "Reims",
]

_FR_POSTAL_PREFIXES = [
    "75", "69", "13", "33", "31", "06", "44", "67", "34", "35",
    "38", "59", "21", "49", "51",
]

_FR_NAME_ROOTS = [
    "Dupont", "Martin", "Bernard", "Durand", "Leblanc",
    "Moreau", "Simon", "Laurent", "Leroy", "Thomas",
    "Conseil", "Services", "Solutions", "Groupe", "Tech",
    "Commerce", "Bâtiment", "Maisons", "Travaux", "Design",
]

_NOISE_PATTERNS = [
    # Casing variants
    lambda s: s.upper(),
    lambda s: s.lower(),
    lambda s: s,  # keep original
    # Abbreviation variants
    lambda s: s.replace("SARL", "S.A.R.L."),
    lambda s: s.replace("SAS", "S.A.S."),
    # Extra comma
    lambda s: s.replace(" ", ", ", 1),
]


class FrenchAugmentor:
    """
    Generates synthetic French-style business records.
    All patterns derived from generic public knowledge of French business naming
    conventions and address formats — no real business data.
    """

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)

    def generate(
        self,
        n_samples: int = 5000,
        source_prefix: str = "SFR",
    ) -> pl.DataFrame:
        """
        Generate `n_samples` synthetic French business records.
        Returns a Polars DataFrame matching the input schema.
        """
        records = []
        for i in range(n_samples):
            name = self._random_name()
            address = self._random_address()
            # Optionally apply noise
            noise_fn = random.choice(_NOISE_PATTERNS)
            noisy_name = noise_fn(name)
            records.append({
                "entity_id": f"{source_prefix}-{i+1:06d}",
                "business_name": noisy_name,
                "business_address": address,
                "country": "France",
                "_synthetic": True,
            })
        df = pl.DataFrame(records)
        logger.info(
            f"FrenchAugmentor: generated {n_samples} synthetic French records."
        )
        return df

    def generate_pairs(
        self,
        n_pairs: int = 1000,
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, list[str]]]:
        """
        Generate source1/source2 pairs with controlled match/no-match labels.
        Returns (source1_df, source2_df, labels_dict).
        """
        source1, source2 = [], []
        labels: dict[str, list[str]] = {}

        for i in range(n_pairs):
            name = self._random_name()
            address = self._random_address()

            s1_id = f"S1-FR-{i+1:05d}"
            s2_id = f"S2-FR-{i+1:05d}"

            source1.append({
                "entity_id": s1_id,
                "business_name": name,
                "business_address": address,
                "country": "France",
            })

            if i % 3 == 0:  # ~33% singletons
                labels[s1_id] = []
            else:
                # Generate noisy variant of same entity
                noise_fn = random.choice(_NOISE_PATTERNS)
                source2.append({
                    "entity_id": s2_id,
                    "business_name": noise_fn(name),
                    "business_address": self._add_address_noise(address),
                    "country": "France",
                })
                labels[s1_id] = [s2_id]

        return (
            pl.DataFrame(source1),
            pl.DataFrame(source2),
            labels,
        )

    # ------------------------------------------------------------------ #
    #  Private generators                                                  #
    # ------------------------------------------------------------------ #

    def _random_name(self) -> str:
        root = random.choice(_FR_NAME_ROOTS)
        suffix = random.choice(_FR_LEGAL_SUFFIXES)
        pattern = random.randint(0, 3)
        if pattern == 0:
            return f"{root} {suffix}"
        elif pattern == 1:
            return f"Société {root} {suffix}"
        elif pattern == 2:
            root2 = random.choice(_FR_NAME_ROOTS)
            return f"{root} & {root2} {suffix}"
        else:
            return f"{root} {random.choice(['et', 'associés', 'partenaires'])} {suffix}"

    def _random_address(self) -> str:
        num = random.randint(1, 999)
        prefix = random.choice(_FR_STREET_PREFIXES)
        root = random.choice(_FR_NAME_ROOTS)
        city = random.choice(_FR_CITIES)
        dept = random.choice(_FR_POSTAL_PREFIXES)
        postal = f"{dept}{random.randint(0, 999):03d}"
        cedex = " Cedex" if random.random() < 0.15 else ""
        return f"{num} {prefix} {root}, {postal} {city}{cedex}, France"

    def _add_address_noise(self, address: str) -> str:
        """Add light noise: abbreviate, reorder, or partially omit."""
        choice = random.randint(0, 2)
        if choice == 0:
            return re.sub(r"Boulevard", "Bd", address)
        elif choice == 1:
            return re.sub(r"Avenue", "Av.", address)
        return address
