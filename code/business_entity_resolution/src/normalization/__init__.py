"""Normalization layer."""
from .normalizer import RecordNormalizer
from .legal_suffix import LegalSuffixCanonicalizer
from .address_parser import AddressParser
from .transliterator import ScriptTransliterator
from .synthetic_france import FrenchAugmentor

__all__ = [
    "RecordNormalizer",
    "LegalSuffixCanonicalizer",
    "AddressParser",
    "ScriptTransliterator",
    "FrenchAugmentor",
]
