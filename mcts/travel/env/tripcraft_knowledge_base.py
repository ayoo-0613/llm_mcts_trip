from __future__ import annotations

import ast
import os
import re
import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _normalize_city(name: str) -> str:
    return str(name).strip().lower()


def _parse_money(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _safe_literal_eval(text: Any) -> Any:
    if text is None:
        return None
    if isinstance(text, (dict, list, tuple)):
        return text
    s = str(text).strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


class TripCraftKnowledgeBase:
    """
    Knowledge base adapter for the local TripCraft database folder.

    It normalizes TripCraft CSV schemas into the lightweight bucket/index
    structures expected by the travel MCTS environment.

    Notes:
    - By default we do NOT ingest the full flights table (often millions of rows).
      The travel environment can still plan using taxi/self-driving via distance matrix.
      Set load_flights=True if you need flight actions.
    """

    def __init__(
        self,
        root: str,
        *,
        load_flights: bool = False,
        keep_raw_frames: bool = True,
    ):
        self.root = root
        self.load_flights = bool(load_flights)
        # TripCraft reference information uses:
        # - taxi cost ~= distance_km
        # - self-driving cost ~= 0.05 * distance_km
        self.ground_cost_per_km = {"taxi": 1.0, "self-driving": 0.05}

        self.states: List[str] = []
        self.cities: List[str] = []
        self.city_to_state: Dict[str, str] = {}
        self.state_to_cities: Dict[str, List[str]] = {}

        self._load_background()
        self._load_tables()
        self._normalize()
        self._build_indexes()
        self._load_transit_index()

        if not keep_raw_frames:
            self.flights = pd.DataFrame()
            self.accommodations = pd.DataFrame()
            self.restaurants = pd.DataFrame()
            self.attractions = pd.DataFrame()
            self.distances = pd.DataFrame()

    # ----------------------------
    # Background (cities/states)
    # ----------------------------
    def _load_background(self) -> None:
        bg_path = os.path.join(self.root, "background", "citySet_with_states_140.txt")
        if not os.path.exists(bg_path):
            raise FileNotFoundError(f"Missing TripCraft background file: {bg_path}")

        city_to_state: Dict[str, str] = {}
        state_to_cities: Dict[str, List[str]] = defaultdict(list)
        with open(bg_path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split("\t")
                if len(parts) < 2:
                    continue
                city, state = parts[0].strip(), parts[1].strip()
                if not city or not state:
                    continue
                city_to_state[city] = state
                state_to_cities[state].append(city)

        self.city_to_state = dict(city_to_state)
        self.state_to_cities = {k: list(v) for k, v in state_to_cities.items()}
        self.states = sorted(self.state_to_cities.keys())
        self.cities = sorted(self.city_to_state.keys())

        # Normalized lookup maps (interface parity with TravelKnowledgeBase).
        self.city_set_norm = {self._normalize_city(c): c for c in self.cities}
        self.state_norm_map = {self._normalize_city(s): s for s in self.states}
        self.city_to_state_norm = {self._normalize_city(c): s for c, s in self.city_to_state.items()}
        self.state_to_cities_norm = {
            self._normalize_city(s): [c for c in cities] for s, cities in self.state_to_cities.items()
        }

    def _normalize_city(self, name: str) -> str:
        return _normalize_city(name)

    def is_state(self, name: str) -> bool:
        norm = self._normalize_city(name)
        return bool(norm and norm in getattr(self, "state_norm_map", {}))

    def cities_in_state(self, state_name: str) -> List[str]:
        norm = self._normalize_city(state_name)
        return list(getattr(self, "state_to_cities_norm", {}).get(norm, []))

    # ----------------------------
    # Tables
    # ----------------------------
    def _load_tables(self) -> None:
        def _read_csv_usecols(path: str, desired: List[str]) -> pd.DataFrame:
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader([f.readline()])
                header = next(reader, [])
            usecols = [c for c in desired if c in header]
            if not usecols:
                # Fallback: let pandas raise a clearer error
                return pd.read_csv(path)
            return pd.read_csv(path, usecols=usecols, low_memory=False)

        # Flights (optional, huge).
        flights_path = os.path.join(self.root, "flights", "cleaned_flights_november_2024.csv")
        if self.load_flights:
            if not os.path.exists(flights_path):
                raise FileNotFoundError(f"Missing flights CSV: {flights_path}")
            self.flights = _read_csv_usecols(
                flights_path,
                [
                    "Flight Number",
                    "Price",
                    "DepTime",
                    "ArrTime",
                    "ActualElapsedTime",
                    "FlightDate",
                    "OriginCityName",
                    "DestCityName",
                    "Distance",
                ],
            )
        else:
            self.flights = pd.DataFrame()

        # Accommodations
        accom_path = os.path.join(self.root, "accommodation", "cleaned_listings_final_v2.csv")
        if not os.path.exists(accom_path):
            raise FileNotFoundError(f"Missing accommodation CSV: {accom_path}")
        self.accommodations = _read_csv_usecols(
            accom_path,
            [
                "City",
                "State",
                "id",
                "name",
                "title",
                "roomType",
                "pricing",
                "max_occupancy",
                "house_rules",
            ],
        )

        # Restaurants
        rest_path = os.path.join(self.root, "restaurants", "cleaned_restaurant_details_2024.csv")
        if not os.path.exists(rest_path):
            raise FileNotFoundError(f"Missing restaurants CSV: {rest_path}")
        self.restaurants = _read_csv_usecols(
            rest_path,
            [
                "id",
                "name",
                "Name",
                "City",
                "cuisines",
                "Cuisines",
                "avg_cost",
                "Average Cost",
                "rating",
                "Aggregate Rating",
                "Rating",
            ],
        )

        # Attractions
        att_path = os.path.join(self.root, "attraction", "cleaned_attractions_final.csv")
        if not os.path.exists(att_path):
            raise FileNotFoundError(f"Missing attractions CSV: {att_path}")
        self.attractions = _read_csv_usecols(
            att_path,
            [
                "id",
                "name",
                "Name",
                "City",
                "State",
                "subcategories",
            ],
        )

        # Distance matrix
        dist_path = os.path.join(self.root, "distance_matrix", "city_distances_times_full.csv")
        if not os.path.exists(dist_path):
            raise FileNotFoundError(f"Missing distance matrix CSV: {dist_path}")
        self.distances = _read_csv_usecols(dist_path, ["origin", "destination", "distance_km"])

    def _normalize(self) -> None:
        # Flights
        if not self.flights.empty:
            for col in ("OriginCityName", "DestCityName"):
                if col in self.flights.columns:
                    self.flights[col] = self.flights[col].astype(str)
            if "Price" in self.flights.columns:
                self.flights["Price"] = pd.to_numeric(self.flights["Price"], errors="coerce")
            self.flights["OriginCityName_norm"] = self.flights["OriginCityName"].apply(self._normalize_city)
            self.flights["DestCityName_norm"] = self.flights["DestCityName"].apply(self._normalize_city)

        # Accommodations
        if not self.accommodations.empty:
            if "City" in self.accommodations.columns:
                self.accommodations["City"] = self.accommodations["City"].astype(str)
                self.accommodations["city_norm"] = self.accommodations["City"].apply(self._normalize_city)
            if "roomType" in self.accommodations.columns:
                self.accommodations["roomType"] = self.accommodations["roomType"].astype(str)

        # Restaurants
        if not self.restaurants.empty:
            if "City" in self.restaurants.columns:
                self.restaurants["City"] = self.restaurants["City"].astype(str)
                self.restaurants["City_norm"] = self.restaurants["City"].apply(self._normalize_city)
            # A few exports use "name" not "Name"
            if "name" in self.restaurants.columns and "Name" not in self.restaurants.columns:
                self.restaurants["Name"] = self.restaurants["name"]
            if "cuisines" in self.restaurants.columns and "Cuisines" not in self.restaurants.columns:
                self.restaurants["Cuisines"] = self.restaurants["cuisines"]
            if "avg_cost" not in self.restaurants.columns:
                # Try to synthesize from a price field if present
                for cand in ("Average Cost", "avgCost", "cost"):
                    if cand in self.restaurants.columns:
                        self.restaurants["avg_cost"] = pd.to_numeric(self.restaurants[cand], errors="coerce")
                        break
            self.restaurants["avg_cost"] = pd.to_numeric(self.restaurants.get("avg_cost"), errors="coerce")
            if "rating" not in self.restaurants.columns:
                for cand in ("Aggregate Rating", "Aggregate_Rating", "Rating"):
                    if cand in self.restaurants.columns:
                        self.restaurants["rating"] = pd.to_numeric(self.restaurants[cand], errors="coerce")
                        break
            self.restaurants["rating"] = pd.to_numeric(self.restaurants.get("rating"), errors="coerce").fillna(0.0)
            self.restaurants["Cuisines_lc"] = self.restaurants.get("Cuisines", "").astype(str).str.lower()

        # Attractions
        if not self.attractions.empty:
            if "City" in self.attractions.columns:
                self.attractions["City"] = self.attractions["City"].astype(str)
                self.attractions["City_norm"] = self.attractions["City"].apply(self._normalize_city)
            if "name" in self.attractions.columns and "Name" not in self.attractions.columns:
                self.attractions["Name"] = self.attractions["name"]
            self.attractions["subcategories_lc"] = self.attractions.get("subcategories", "").astype(str).str.lower()

        # Distances
        if not self.distances.empty:
            for col in ("origin", "destination"):
                if col in self.distances.columns:
                    self.distances[col] = self.distances[col].astype(str)
            self.distances["origin_norm"] = self.distances["origin"].apply(self._normalize_city)
            self.distances["destination_norm"] = self.distances["destination"].apply(self._normalize_city)
            if "distance_km" in self.distances.columns:
                self.distances["distance_km"] = pd.to_numeric(self.distances["distance_km"], errors="coerce")

    def _build_indexes(self) -> None:
        # Distance existence & quick km lookup.
        self._distance_pairs = set()
        self._distance_km_map: Dict[Tuple[str, str], float] = {}
        if not self.distances.empty and "origin_norm" in self.distances.columns and "destination_norm" in self.distances.columns:
            for _, row in self.distances.iterrows():
                orig = row.get("origin_norm")
                dest = row.get("destination_norm")
                if pd.isna(orig) or pd.isna(dest):
                    continue
                key = (str(orig), str(dest))
                self._distance_pairs.add(key)
                km = row.get("distance_km")
                try:
                    if km is not None and not pd.isna(km):
                        self._distance_km_map[key] = float(km)
                except Exception:
                    pass

        # Flights (optional): build a light per-pair bucket list if loaded.
        self._flight_pairs = set()
        self._flight_buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        if not self.flights.empty:
            # Filter flights to TripCraft city set to keep memory bounded.
            allowed = set(self.city_set_norm.keys())
            df = self.flights
            df = df[df["OriginCityName_norm"].isin(allowed) & df["DestCityName_norm"].isin(allowed)]

            for _, row in df.iterrows():
                orig = row.get("OriginCityName_norm")
                dest = row.get("DestCityName_norm")
                if pd.isna(orig) or pd.isna(dest):
                    continue
                pair = (str(orig), str(dest))
                self._flight_pairs.add(pair)
                self._flight_buckets.setdefault(pair, []).append(
                    {
                        "id": row.get("Flight Number"),
                        "price": float(row.get("Price")) if row.get("Price") is not None and not pd.isna(row.get("Price")) else None,
                        "origin": row.get("OriginCityName"),
                        "destination": row.get("DestCityName"),
                        "depart": row.get("DepTime"),
                        "arrive": row.get("ArrTime"),
                        "duration": row.get("ActualElapsedTime"),
                        "date": row.get("FlightDate"),
                        "distance": row.get("Distance"),
                    }
                )
            for pair, items in self._flight_buckets.items():
                items.sort(key=lambda f: (f.get("price", float("inf")), f.get("duration", float("inf"))))

        # Accommodations by city.
        accom_buckets: Dict[str, List[Dict[str, Any]]] = {}
        for idx, row in self.accommodations.iterrows():
            city_norm = row.get("city_norm")
            if pd.isna(city_norm):
                continue

            pricing = _safe_literal_eval(row.get("pricing"))
            nightly_price = None
            if isinstance(pricing, dict):
                nightly_price = _parse_money(pricing.get("price") or pricing.get("originalPrice") or pricing.get("total"))
            if nightly_price is None:
                nightly_price = _parse_money(row.get("pricing"))

            room_type_raw = str(row.get("roomType") or "").strip().lower()
            room_type = room_type_raw
            # Map TripCraft roomType to TravelPlanner-like names used by our constraint matcher.
            if "shared" in room_type_raw:
                room_type = "Shared room"
            elif "private" in room_type_raw:
                room_type = "Private room"
            elif "entire" in room_type_raw or "home" in room_type_raw:
                room_type = "Entire home/apt"

            raw_rules = row.get("house_rules")
            if isinstance(raw_rules, str):
                cleaned = raw_rules.replace("&", ",").replace(" and ", ",")
                tokens = {tok.strip().lower().replace(" ", "_") for tok in cleaned.split(",") if tok.strip()}
            else:
                tokens = set()

            stay = {
                "id": str(row.get("id") if row.get("id") is not None else idx),
                "name": str(row.get("name") or row.get("title") or "").strip(),
                "price": float(nightly_price) if nightly_price is not None else None,
                "minimum_nights": None,
                "room_type": room_type,
                "review": None,
                "occupancy": row.get("max_occupancy"),
                "city": row.get("City"),
                "house_rules": raw_rules,
                "house_rules_tokens": tokens,
            }
            accom_buckets.setdefault(str(city_norm), []).append(stay)

        for city_norm, items in accom_buckets.items():
            items.sort(key=lambda s: (s.get("price") if s.get("price") is not None else float("inf")))
        self._accommodation_buckets = accom_buckets

        # Restaurants by city.
        rest_buckets: Dict[str, List[Dict[str, Any]]] = {}
        for idx, row in self.restaurants.iterrows():
            city_norm = row.get("City_norm")
            if pd.isna(city_norm):
                continue
            restaurant = {
                "id": str(row.get("id") if row.get("id") is not None else idx),
                "name": str(row.get("Name") or "").strip(),
                "city": str(row.get("City") or "").strip(),
                "cuisines": str(row.get("Cuisines") or "").strip(),
                "cuisines_lc": str(row.get("Cuisines_lc") or "").strip(),
                "cost": float(row.get("avg_cost")) if row.get("avg_cost") is not None and not pd.isna(row.get("avg_cost")) else None,
                "rating": float(row.get("rating") or 0.0),
            }
            rest_buckets.setdefault(str(city_norm), []).append(restaurant)
        for city_norm, items in rest_buckets.items():
            items.sort(key=lambda r: (-r.get("rating", 0.0), r.get("cost", float("inf"))))
        self._restaurant_buckets = rest_buckets

        # Attractions by city.
        att_buckets: Dict[str, List[Dict[str, Any]]] = {}
        for idx, row in self.attractions.iterrows():
            city_norm = row.get("City_norm")
            if pd.isna(city_norm):
                continue
            att = {
                "id": str(row.get("id") if row.get("id") is not None else idx),
                "name": str(row.get("Name") or "").strip(),
                "city": str(row.get("City") or "").strip(),
                "subcategories": row.get("subcategories"),
                "subcategories_lc": str(row.get("subcategories_lc") or "").strip(),
            }
            att_buckets.setdefault(str(city_norm), []).append(att)
        self._attraction_buckets = att_buckets

    # ----------------------------
    # Transit PoI nearest stop index (optional)
    # ----------------------------
    def _load_transit_index(self) -> None:
        self._poi_transit: Dict[Tuple[str, str], Dict[str, Any]] = {}
        path = os.path.join(self.root, "public_transit_gtfs", "all_poi_nearest_stops.csv")
        if not os.path.exists(path):
            return
        df = pd.read_csv(
            path,
            usecols=["City", "PoI", "nearest_stop_name", "nearest_stop_distance"],
            low_memory=False,
        )
        for _, row in df.iterrows():
            city = row.get("City")
            poi = row.get("PoI")
            if pd.isna(city) or pd.isna(poi):
                continue
            key = (self._normalize_city(str(city)), self._normalize_city(str(poi)))
            self._poi_transit[key] = {
                "nearest_stop_name": row.get("nearest_stop_name"),
                "nearest_stop_distance": row.get("nearest_stop_distance"),
            }

    def nearest_transit(self, *, city: str, poi: str) -> Optional[Dict[str, Any]]:
        if not city or not poi:
            return None
        return self._poi_transit.get((self._normalize_city(city), self._normalize_city(poi)))

    # ----------------------------
    # Unified interface used by env/retrieval
    # ----------------------------
    def get_attractions(self, city: str, top_k: int = 5) -> List[Dict[str, Any]]:
        city_norm = self._normalize_city(city)
        return list(self._attraction_buckets.get(city_norm, [])[: int(top_k)])

    def has_any_transport(self, origin: str, destination: str, require_flight: bool = False) -> bool:
        orig = self._normalize_city(origin)
        dest = self._normalize_city(destination)
        if require_flight:
            return (orig, dest) in getattr(self, "_flight_pairs", set())
        return (orig, dest) in getattr(self, "_distance_pairs", set()) or (orig, dest) in getattr(self, "_flight_pairs", set())

    def distance_km(self, origin: str, destination: str) -> Optional[float]:
        if not origin or not destination:
            return None
        key = (self._normalize_city(origin), self._normalize_city(destination))
        if key in self._distance_km_map:
            return float(self._distance_km_map[key])
        return None
