"""
Geographic Blocker — groups records by normalized geo bucket.

Bucket = (country_normalized, city_normalized) or (country, postal_prefix).
This is a hard, cheap pre-filter that dramatically shrinks the candidate space
for geographically close entities without any external geocoding service.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

import polars as pl
from loguru import logger


def _normalize_country(country: str) -> str:
    """Normalize country string to a canonical token."""
    if not country:
        return "unknown"
    c = country.strip().lower()
    # Aliases
    aliases = {
        "united states": "us", "united states of america": "us",
        "usa": "us", "u.s.a.": "us", "u.s.": "us",
        "india": "in", "bharat": "in",
        "france": "fr", "fr": "fr", "french republic": "fr",
    }
    return aliases.get(c, c[:2])  # fallback: first 2 chars


def _normalize_city(city: str) -> str:
    if not city:
        return ""
    return re.sub(r"[^a-z0-9]", "", city.lower())


class GeoBlocker:
    """
    Assigns entities to geographic buckets and retrieves candidates
    sharing the same bucket as a query.
    """

    def build_index(
        self,
        ids: list[str],
        countries: list[str],
        cities: list[str],
        postals: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        """
        Build {geo_bucket: [entity_ids]} index.
        Bucket strategies:
          1. (country, city) — preferred when city is available
          2. (country, postal_prefix) — fallback when city missing
          3. (country,) — last resort
        """
        index: dict[str, list[str]] = defaultdict(list)
        for i, eid in enumerate(ids):
            norm_country = _normalize_country(countries[i] if i < len(countries) else "")
            norm_city = _normalize_city(cities[i] if i < len(cities) else "")
            postal = (postals[i] if postals and i < len(postals) else "") or ""
            postal_prefix = postal[:3] if postal else ""

            # Bucket 1: country + city
            if norm_city:
                index[f"{norm_country}::{norm_city}"].append(eid)

            # Bucket 2: country + postal prefix
            if postal_prefix:
                index[f"{norm_country}::zip:{postal_prefix}"].append(eid)

            # Bucket 3: country only (broad)
            index[f"{norm_country}"].append(eid)

        return dict(index)

    def get_candidates(
        self,
        query_country: str,
        query_city: str,
        query_postal: str,
        geo_index: dict[str, list[str]],
        exclude_id: Optional[str] = None,
    ) -> list[str]:
        """Return candidate IDs in the same geographic bucket."""
        norm_country = _normalize_country(query_country)
        norm_city = _normalize_city(query_city)
        postal_prefix = (query_postal or "")[:3]

        cands = set()
        if norm_city:
            cands.update(geo_index.get(f"{norm_country}::{norm_city}", []))
        if postal_prefix:
            cands.update(geo_index.get(f"{norm_country}::zip:{postal_prefix}", []))
        # If few candidates, broaden to country level
        if len(cands) < 5:
            cands.update(geo_index.get(norm_country, []))

        if exclude_id:
            cands.discard(exclude_id)
        return list(cands)
