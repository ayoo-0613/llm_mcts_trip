from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple, Set
import os
import pandas as pd

from tools.accommodations.apis import Accommodations
from tools.attractions.apis import Attractions
from tools.flights.apis import Flights
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.restaurants.apis import Restaurants

class TravelKnowledgeBase:
    def __init__(self, root: str = "database", *, keep_raw_frames: bool = True):
        self.root = root
        self.flights = self._load_csv("flights/clean_Flights_2022.csv")
        self.accommodations = self._load_csv("accommodations/clean_accommodations_2022.csv")
        self.restaurants = self._load_csv("restaurants/clean_restaurant_2022.csv")
        self.attractions = self._load_csv("attractions/attractions.csv")
        self.distances = self._load_csv("googleDistanceMatrix/distance.csv")
        self.states, self.cities, self.city_to_state, self.state_to_cities = self._load_background()

        self._normalize()
        self._build_indexes()
        if not keep_raw_frames:
            # Indexes are built from the raw DataFrames into lightweight bucket structures;
            # drop the big frames to reduce peak RSS in long-running experiments.
            self.flights = pd.DataFrame()
            self.accommodations = pd.DataFrame()
            self.restaurants = pd.DataFrame()
            self.attractions = pd.DataFrame()

    # ----------------------------
    # State / city helpers
    # ----------------------------
    def is_state(self, name: str) -> bool:
        """Return True if the given name matches a known state."""
        norm = self._normalize_city(name)
        return bool(norm and norm in getattr(self, "state_norm_map", {}))

    def cities_in_state(self, state_name: str) -> List[str]:
        """Return all cities recorded for a given state (original casing)."""
        norm = self._normalize_city(state_name)
        return list(getattr(self, "state_to_cities_norm", {}).get(norm, []))

    def _load_csv(self, relative_path: str) -> pd.DataFrame:
        path = os.path.join(self.root, relative_path)
        if relative_path.startswith("flights/"):
            return Flights(csv_path=path).data
        if relative_path.startswith("accommodations/"):
            return Accommodations(csv_path=path).data
        if relative_path.startswith("restaurants/"):
            return Restaurants(csv_path=path).data
        if relative_path.startswith("attractions/"):
            return Attractions(csv_path=path).data
        if relative_path.startswith("googleDistanceMatrix/"):
            return GoogleDistanceMatrix(csv_path=path).data
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected dataset at {path}")
        return pd.read_csv(path)

    def _load_background(self):
        """Load supporting background data: states, cities, and city-state pairs."""
        bg_root = os.path.join(self.root, "background")
        states = self._load_txt_lines(os.path.join(bg_root, "stateSet.txt"))
        cities = self._load_txt_lines(os.path.join(bg_root, "citySet.txt"))
        city_to_state = self._load_city_state_map(os.path.join(bg_root, "citySet_with_states.txt"))

        state_to_cities: Dict[str, List[str]] = defaultdict(list)
        for city, state in city_to_state.items():
            if state:
                state_to_cities[state].append(city)

        return states, cities, city_to_state, dict(state_to_cities)

    @staticmethod
    def _load_txt_lines(path: str) -> List[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected dataset at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _load_city_state_map(path: str) -> Dict[str, str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected dataset at {path}")
        mapping: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split("\t")
                if len(parts) < 2:
                    continue
                city, state = parts[0].strip(), parts[1].strip()
                if city:
                    mapping[city] = state
        return mapping

    @staticmethod
    def _normalize_city(name: str) -> str:
        return str(name).strip().lower()

    def _normalize(self) -> None:
        if "Price" in self.flights:
            self.flights["Price"] = pd.to_numeric(self.flights["Price"], errors="coerce")
        self.flights["OriginCityName_norm"] = self.flights["OriginCityName"].apply(self._normalize_city)
        self.flights["DestCityName_norm"] = self.flights["DestCityName"].apply(self._normalize_city)

        if "price" in self.accommodations:
            self.accommodations["price"] = pd.to_numeric(self.accommodations["price"], errors="coerce")
        self.accommodations["city_norm"] = self.accommodations["city"].apply(self._normalize_city)

        if "Average Cost" in self.restaurants:
            self.restaurants["Average Cost"] = pd.to_numeric(self.restaurants["Average Cost"], errors="coerce")
        self.restaurants["City_norm"] = self.restaurants["City"].apply(self._normalize_city)
        self.restaurants["Cuisines_lc"] = self.restaurants["Cuisines"].fillna("").str.lower()

        self.attractions["City_norm"] = self.attractions["City"].apply(self._normalize_city)

        if "origin" in self.distances:
            self.distances["origin_norm"] = self.distances["origin"].apply(self._normalize_city)
            self.distances["destination_norm"] = self.distances["destination"].apply(self._normalize_city)

        # Background normalization for quick lookups.
        self.state_norm_map = {
            self._normalize_city(state): state for state in self.states if state
        }
        self.city_to_state_norm = {
            self._normalize_city(city): state for city, state in self.city_to_state.items() if city
        }
        self.state_to_cities_norm = {
            self._normalize_city(state): list(cities) for state, cities in self.state_to_cities.items()
        }
        self.city_set_norm = {
            self._normalize_city(city): city for city in self.cities if city
        }

    def _build_indexes(self) -> None:
        """Pre-index heavy tables to avoid repeated DataFrame scans during planning."""
        self._flight_pairs: Set[Tuple[str, str]] = set()
        self._distance_pairs: Set[Tuple[str, str]] = set()
        self._flight_buckets: Dict[Tuple[str, str], List[Dict]] = {}
        self._accommodation_buckets: Dict[str, List[Dict]] = {}
        self._restaurant_buckets: Dict[str, List[Dict]] = {}
        self._attraction_buckets: Dict[str, List[Dict]] = {}

        # Flights by (origin, dest), pre-sorted by price then duration.
        flight_buckets: Dict[Tuple[str, str], List[Dict]] = {}
        for _, row in self.flights.iterrows():
            orig = row.get("OriginCityName_norm")
            dest = row.get("DestCityName_norm")
            if pd.isna(orig) or pd.isna(dest):
                continue
            pair = (orig, dest)
            self._flight_pairs.add(pair)
            flight_buckets.setdefault(pair, []).append(
                {
                    "id": row["Flight Number"],
                    "price": float(row["Price"]),
                    "origin": row["OriginCityName"],
                    "destination": row["DestCityName"],
                    "depart": row["DepTime"],
                    "arrive": row["ArrTime"],
                    "duration": row["ActualElapsedTime"],
                    "date": row["FlightDate"],
                    "distance": row.get("Distance"),
                }
            )
        for pair, items in flight_buckets.items():
            items.sort(key=lambda f: (f.get("price", float("inf")), f.get("duration", float("inf"))))
        self._flight_buckets = flight_buckets

        # Distance matrix pairs for quick existence checks.
        if "origin_norm" in self.distances and "destination_norm" in self.distances:
            for _, row in self.distances.iterrows():
                orig = row.get("origin_norm")
                dest = row.get("destination_norm")
                if pd.isna(orig) or pd.isna(dest):
                    continue
                self._distance_pairs.add((orig, dest))

        # Accommodations by city, pre-sorted by price asc, review desc.
        accom_buckets: Dict[str, List[Dict]] = {}
        for idx, row in self.accommodations.iterrows():
            city_norm = row.get("city_norm")
            if pd.isna(city_norm):
                continue
            raw_rules = row.get("house_rules")
            if isinstance(raw_rules, str):
                cleaned = raw_rules.replace("&", ",").replace(" and ", ",")
                tokens = {tok.strip().lower().replace(" ", "_") for tok in cleaned.split(",") if tok}
            elif isinstance(raw_rules, list):
                tokens = {str(tok).strip().lower().replace(" ", "_") for tok in raw_rules if tok}
            else:
                tokens = set()
            stay = {
                "id": str(idx),
                "name": row["NAME"],
                "price": float(row["price"]) if not pd.isna(row["price"]) else None,
                "minimum_nights": None,
                "room_type": row["room type"],
                "review": row.get("review rate number"),
                "occupancy": row.get("maximum occupancy"),
                "city": row["city"],
                "house_rules": row.get("house_rules"),
                "house_rules_tokens": tokens,
            }
            mn = row.get("minimum nights")
            try:
                if mn is not None and not pd.isna(mn):
                    stay["minimum_nights"] = int(float(mn))
            except Exception:
                stay["minimum_nights"] = None
            accom_buckets.setdefault(city_norm, []).append(stay)
        def _review_score(val: Any) -> float:
            try:
                return float(val)
            except Exception:
                return 0.0
        for city_norm, items in accom_buckets.items():
            items.sort(
                key=lambda s: (
                    s.get("price") if s.get("price") is not None else float("inf"),
                    -_review_score(s.get("review")),
                )
            )
        self._accommodation_buckets = accom_buckets

        # Restaurants by city, pre-sorted by rating desc then cost asc.
        rest_buckets: Dict[str, List[Dict]] = {}
        for idx, row in self.restaurants.iterrows():
            city_norm = row.get("City_norm")
            if pd.isna(city_norm):
                continue
            restaurant = {
                "id": str(idx),
                "name": row["Name"],
                "city": row["City"],
                "cuisines": row["Cuisines"],
                "cuisines_lc": row["Cuisines_lc"],
                "cost": float(row["Average Cost"]) if not pd.isna(row["Average Cost"]) else None,
                "rating": float(row["Aggregate Rating"]) if not pd.isna(row["Aggregate Rating"]) else 0.0,
            }
            rest_buckets.setdefault(city_norm, []).append(restaurant)
        for city_norm, items in rest_buckets.items():
            items.sort(key=lambda r: (-r.get("rating", 0.0), r.get("cost", float("inf"))))
        self._restaurant_buckets = rest_buckets

        # Attractions by city (order preserved as in CSV).
        att_buckets: Dict[str, List[Dict]] = {}
        for idx, row in self.attractions.iterrows():
            city_norm = row.get("City_norm")
            if pd.isna(city_norm):
                continue
            att = {
                "id": str(idx),
                "name": row["Name"],
                "address": row.get("Address"),
                "phone": row.get("Phone"),
                "website": row.get("Website"),
                "city": row["City"],
            }
            att_buckets.setdefault(city_norm, []).append(att)
        self._attraction_buckets = att_buckets

    def get_flights(self, origin: str, destination: str, top_k: int = 5,
                    max_price: Optional[float] = None) -> List[Dict]:
        orig = self._normalize_city(origin)
        dest = self._normalize_city(destination)
        flights = self._flight_buckets.get((orig, dest), [])
        if max_price is not None:
            flights = [f for f in flights if f.get("price") is not None and f["price"] <= max_price]
        return flights[:top_k]

    def get_accommodations(self, city: str, top_k: int = 5,
                           max_price: Optional[float] = None,
                           house_rules: Optional[List[str]] = None,
                           min_occupancy: Optional[int] = None) -> List[Dict]:
        city_norm = self._normalize_city(city)
        stays = self._accommodation_buckets.get(city_norm, [])
        if max_price is not None:
            stays = [s for s in stays if s.get("price") is not None and s["price"] <= max_price]
        if min_occupancy is not None:
            stays = [
                s for s in stays
                if s.get("occupancy") is None or s.get("occupancy") >= min_occupancy
            ]
        if house_rules:
            rules = {str(r).strip().lower().replace(" ", "_") for r in house_rules if r}
            stays = [
                s for s in stays
                if rules.issubset(s.get("house_rules_tokens") or set())
            ]
        return stays[:top_k]

    def get_restaurants(self, city: str, preferences: Optional[List[str]] = None,
                        top_k: int = 5) -> List[Dict]:
        city_norm = self._normalize_city(city)
        restaurants = self._restaurant_buckets.get(city_norm, [])
        prefs = [str(p).strip().lower() for p in preferences] if preferences else []
        if prefs:
            restaurants = [
                r for r in restaurants
                if any(pref and pref in r.get("cuisines_lc", "") for pref in prefs)
            ]
        restaurants = restaurants[:top_k]
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "city": r["city"],
                "cuisines": r["cuisines"],
                "cost": r["cost"] if r.get("cost") is not None else 0.0,
                "rating": r["rating"],
            }
            for r in restaurants
        ]

    def get_attractions(self, city: str, top_k: int = 5) -> List[Dict]:
        city_norm = self._normalize_city(city)
        attractions = self._attraction_buckets.get(city_norm, [])
        return attractions[:top_k]

    def get_state_for_city(self, city: str) -> Optional[str]:
        """Return the state a city belongs to, if known."""
        return self.city_to_state_norm.get(self._normalize_city(city))

    def get_cities_for_state(self, state: str) -> List[str]:
        """Return all cities recorded for a given state."""
        norm = self._normalize_city(state)
        canonical = self.state_norm_map.get(norm)
        if not canonical:
            return []
        return list(self.state_to_cities.get(canonical, []))

    def expand_locations_to_cities(self, locations: List[str]) -> List[str]:
        """Expand mixed city/state names into a unique city list."""
        expanded: List[str] = []
        seen = set()
        for loc in locations or []:
            norm = self._normalize_city(loc)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            # If it's a state, add all its cities; otherwise keep as-is
            cities = self.get_cities_for_state(loc)
            if cities:
                for city in cities:
                    city_norm = self._normalize_city(city)
                    if city_norm not in seen:
                        seen.add(city_norm)
                        expanded.append(city)
            else:
                expanded.append(loc)
        return expanded

    def has_any_transport(self, origin: str, destination: str, require_flight: bool = False) -> bool:
        """
        Check if any transport data exists between two locations.
        If require_flight=True, only consider flights; otherwise allow distance matrix fallback.
        """
        orig = self._normalize_city(origin)
        dest = self._normalize_city(destination)
        if (orig, dest) in self._flight_pairs:
            return True
        if require_flight:
            return False
        if self._distance_pairs and (orig, dest) in self._distance_pairs:
            return True
        return False

    def get_candidate_cities(self, destination_hint: Optional[str] = None,
                             must_visit: Optional[List[str]] = None,
                             priority: Optional[List[str]] = None,
                             top_k: int = 10) -> List[str]:
        must_visit = must_visit or []
        priority = priority or []
        dest_norm = self._normalize_city(destination_hint) if destination_hint else None
        state_hint = None
        if destination_hint:
            # If the hint is a state or maps to one through the background sets, remember it.
            state_hint = self.state_norm_map.get(dest_norm) or self.city_to_state_norm.get(dest_norm)

        def _add_unique(src: List[str], acc: List[str], seen: set) -> None:
            for city in src:
                norm = self._normalize_city(city)
                if norm and norm not in seen:
                    seen.add(norm)
                    acc.append(city)

        candidates: List[str] = []
        seen = set()
        _add_unique(must_visit, candidates, seen)
        _add_unique(priority, candidates, seen)
        if destination_hint:
            _add_unique([destination_hint], candidates, seen)

        if state_hint:
            _add_unique(self.get_cities_for_state(state_hint), candidates, seen)

        if destination_hint:
            accom_matches = self.accommodations[
                self.accommodations["city"].str.contains(destination_hint, case=False, na=False)
            ]["city"].unique().tolist()
            flight_matches = self.flights[
                self.flights["DestCityName"].str.contains(destination_hint, case=False, na=False) |
                self.flights["OriginCityName"].str.contains(destination_hint, case=False, na=False)
            ]["DestCityName"].unique().tolist()
            _add_unique(accom_matches, candidates, seen)
            _add_unique(flight_matches, candidates, seen)

        if len(candidates) < top_k and self.cities:
            _add_unique(self.cities, candidates, seen)

        # Fallback to most frequent accommodation cities to fill up to top_k
        if len(candidates) < top_k:
            counts = self.accommodations["city"].value_counts()
            for city in counts.index:
                _add_unique([city], candidates, seen)
                if len(candidates) >= top_k:
                    break
        return candidates[:top_k]

    def distance_between(self, origin: str, destination: str) -> Optional[str]:
        orig = self._normalize_city(origin)
        dest = self._normalize_city(destination)
        df = self.distances[
            (self.distances["origin_norm"] == orig) &
            (self.distances["destination_norm"] == dest)
        ]
        if df.empty:
            return None
        row = df.iloc[0]
        return str(row.get("distance"))

    def distance_km(self, origin: str, destination: str) -> Optional[float]:
        raw = self.distance_between(origin, destination)
        if raw is None:
            return None
        try:
            import re
            # 1) 先把千位分隔符去掉
            s = str(raw).replace(",", "")
            # 2) 再从中提取第一个数字（可带小数）
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
            if not match:
                return None
            return float(match.group(1))
        except Exception:
            return None

    # ----------------------------
    # Unified slot queries
    # ----------------------------
    @staticmethod
    def _slot_attr(slot: Any, name: str):
        if slot is None:
            return None
        if hasattr(slot, name):
            return getattr(slot, name)
        if isinstance(slot, dict):
            return slot.get(name)
        return None

    @staticmethod
    def _time_to_minutes(val: Any) -> Optional[int]:
        try:
            text = str(val)
            if not text or text.lower() == "nan":
                return None
            parts = text.replace(".", ":").split(":")
            if len(parts) < 2:
                return None
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return None

    @staticmethod
    def _used_restaurant_ids(state: Any) -> Set[str]:
        ids: Set[str] = set()
        if not state or not hasattr(state, "meals"):
            return ids
        try:
            for day_map in state.meals.values():
                for meal in day_map.values():
                    if meal and "id" in meal:
                        ids.add(str(meal["id"]))
        except Exception:
            return ids
        return ids

    @staticmethod
    def _used_attraction_ids(state: Any) -> Set[str]:
        ids: Set[str] = set()
        if not state or not hasattr(state, "attractions"):
            return ids
        try:
            for day_map in state.attractions.values():
                for att in day_map.values():
                    if att and "id" in att:
                        ids.add(str(att["id"]))
        except Exception:
            return ids
        return ids

    @staticmethod
    def _apply_numeric_cap(df, col: str, cap):
        """Hard filter dataframe by numeric cap; cap<=0 yields empty."""
        if cap is None:
            return df
        try:
            cap_v = float(cap)
        except Exception:
            return df
        if cap_v <= 0:
            return df.iloc[0:0]
        if col in df.columns:
            return df[df[col].notna() & (df[col].astype(float) <= cap_v)]
        return df

    def query(self, slot, filt: Dict[str, Any], state: Any, cap: int = 80) -> List[Any]:
        stype = self._slot_attr(slot, "type")
        if not stype:
            return []
        stype = str(stype).lower()
        if stype in ("segment", "flight"):
            return self.query_flights(slot, filt, state, cap=cap)
        if stype == "hotel":
            return self.query_hotels(slot, filt, state, cap=cap)
        if stype == "meal" or stype == "restaurant":
            return self.query_restaurants(slot, filt, state, cap=cap)
        if stype == "attraction":
            return self.query_attractions(slot, filt, state, cap=cap)
        if stype == "finish":
            return ["finish"]
        if stype == "city":
            return []
        return []

    def _flight_sort_key(self, item: Dict[str, Any], idx: int, sort_by: str):
        price = item.get("price") if item.get("price") is not None else float("inf")
        duration = item.get("duration") if item.get("duration") is not None else float("inf")
        depart_m = self._time_to_minutes(item.get("depart")) or float("inf")
        arrive_m = self._time_to_minutes(item.get("arrive")) or float("inf")
        if sort_by == "duration":
            return (duration, price, depart_m, idx)
        if sort_by == "depart":
            return (depart_m, price, duration, idx)
        if sort_by == "arrive":
            return (arrive_m, price, duration, idx)
        return (price, duration, depart_m, idx)

    def query_flights(self, slot, filt: Dict[str, Any], state: Any, cap: int = 80) -> List[Dict]:
        origin = self._slot_attr(slot, "origin") or self._slot_attr(slot, "from_city") or self._slot_attr(slot, "from")
        destination = (
            self._slot_attr(slot, "destination") or self._slot_attr(slot, "to_city") or self._slot_attr(slot, "to")
        )
        if not origin or not destination:
            return []
        orig_norm = self._normalize_city(origin)
        dest_norm = self._normalize_city(destination)
        flights = list(self._flight_buckets.get((orig_norm, dest_norm), []))
        avoid_ids = {str(x) for x in (filt.get("avoid_ids") or [])}

        max_price = filt.get("max_price")
        min_price = filt.get("min_price")
        earliest = self._time_to_minutes(filt.get("earliest_depart"))
        latest = self._time_to_minutes(filt.get("latest_depart"))
        max_duration = filt.get("max_duration")
        max_stops = filt.get("max_stops")
        filter_date = filt.get("date")

        df = pd.DataFrame(flights)
        if not df.empty:
            df = df[~df["id"].astype(str).isin(avoid_ids)]
            df = self._apply_numeric_cap(df, "price", max_price)
            if min_price is not None and "price" in df.columns:
                try:
                    df = df[df["price"].notna() & (df["price"].astype(float) >= float(min_price))]
                except Exception:
                    pass
            if max_duration is not None:
                try:
                    df = df[(df["duration"].notna()) & (df["duration"].astype(float) <= float(max_duration))]
                except Exception:
                    pass
            if max_stops is not None and "stops" in df.columns:
                try:
                    df = df[(df["stops"].notna()) & (df["stops"].astype(float) <= float(max_stops))]
                except Exception:
                    pass
            if filter_date is not None and "date" in df.columns:
                df = df[df["date"].astype(str) == str(filter_date)]
            # time window filters
            if earliest is not None:
                df = df[df["depart"].apply(lambda t: self._time_to_minutes(t) is not None and self._time_to_minutes(t) >= earliest)]
            if latest is not None:
                df = df[df["depart"].apply(lambda t: self._time_to_minutes(t) is not None and self._time_to_minutes(t) <= latest)]

        filtered: List[Tuple[Dict, int]] = []
        for idx, row in df.reset_index(drop=True).iterrows() if not df.empty else []:
            filtered.append((row.to_dict(), idx))

        if not filtered:
            return []
        sort_by = str(filt.get("sort_by") or "price").lower()
        filtered.sort(key=lambda pair: self._flight_sort_key(pair[0], pair[1], sort_by))
        candidates = [item for item, _ in filtered[:cap]]
        if max_price is not None:
            try:
                mp = float(max_price)
                bad = [c for c in candidates if c.get("price") is not None and float(c["price"]) > mp + 1e-6]
                if bad:
                    raise RuntimeError(
                        f"[KB] flight cap violated max_price={mp} bad0_price={bad[0].get('price')} bad0={bad[0]}"
                    )
            except Exception:
                pass
        return candidates

    def _hotel_sort_key(self, item: Dict[str, Any], idx: int, sort_by: str):
        price = item.get("price") if item.get("price") is not None else float("inf")
        review = item.get("review") if item.get("review") is not None else 0.0
        if sort_by == "review":
            return (-review, price, idx)
        return (price, -review, idx)

    def query_hotels(self, slot, filt: Dict[str, Any], state: Any, cap: int = 80) -> List[Dict]:
        city = self._slot_attr(slot, "city") or self._slot_attr(slot, "destination")
        if not city:
            return []
        city_norm = self._normalize_city(city)
        stays = list(self._accommodation_buckets.get(city_norm, []))
        avoid_ids = {str(x) for x in (filt.get("avoid_ids") or [])}

        max_price = filt.get("max_price")
        min_price = filt.get("min_price")
        min_review = filt.get("min_review")
        max_minimum_nights = filt.get("max_minimum_nights") or filt.get("max_min_nights")
        room_types = {s.lower() for s in filt.get("room_type") or []}
        house_rules = {str(s).lower().replace(" ", "_") for s in filt.get("house_rules") or []}
        deny_rules = {r for r in house_rules if r.startswith("no_")}
        allow_rules = {r for r in house_rules if not r.startswith("no_")}
        forbid_tokens = {f"no_{r}" for r in allow_rules}
        min_occupancy = filt.get("min_occupancy")

        df = pd.DataFrame(stays)
        if not df.empty:
            df = df[~df["id"].astype(str).isin(avoid_ids)]
            df = self._apply_numeric_cap(df, "price", max_price)
            if min_price is not None and "price" in df.columns:
                try:
                    df = df[df["price"].notna() & (df["price"].astype(float) >= float(min_price))]
                except Exception:
                    pass
            if min_review is not None and "review" in df.columns:
                try:
                    df = df[df["review"].notna() & (df["review"].astype(float) >= float(min_review))]
                except Exception:
                    pass
            if max_minimum_nights is not None and "minimum_nights" in df.columns:
                try:
                    cap_n = float(max_minimum_nights)
                    df = df[
                        df["minimum_nights"].apply(
                            lambda v: v is None or pd.isna(v) or float(v) <= cap_n
                        )
                    ]
                except Exception:
                    pass
            if room_types:
                df = df[df["room_type"].apply(lambda x: isinstance(x, str) and x.lower() in room_types)]
            if min_occupancy is not None and "occupancy" in df.columns:
                try:
                    df = df[
                        df["occupancy"].apply(
                            lambda v: v is None or (not pd.isna(v) and float(v) >= float(min_occupancy))
                        )
                    ]
                except Exception:
                    pass
            if deny_rules or forbid_tokens:
                def _tok_set(tokens: Any) -> set:
                    if isinstance(tokens, set):
                        return tokens
                    if isinstance(tokens, (list, tuple)):
                        return set(tokens)
                    return set()
                if deny_rules:
                    df = df[df["house_rules_tokens"].apply(lambda tokens: deny_rules.issubset(_tok_set(tokens)))]
                if forbid_tokens:
                    df = df[df["house_rules_tokens"].apply(lambda tokens: forbid_tokens.isdisjoint(_tok_set(tokens)))]

        filtered: List[Tuple[Dict, int]] = []
        for idx, row in df.reset_index(drop=True).iterrows() if not df.empty else []:
            filtered.append((row.to_dict(), idx))

        if not filtered:
            return []
        sort_by = str(filt.get("sort_by") or "price").lower()
        filtered.sort(key=lambda pair: self._hotel_sort_key(pair[0], pair[1], sort_by))
        candidates = [item for item, _ in filtered[:cap]]
        if max_price is not None:
            try:
                mp = float(max_price)
                bad = [c for c in candidates if c.get("price") is not None and float(c["price"]) > mp + 1e-6]
                if bad:
                    raise RuntimeError(
                        f"[KB] hotel cap violated max_price={mp} bad0_price={bad[0].get('price')} bad0={bad[0]}"
                    )
            except Exception:
                pass
        return candidates

    def _restaurant_sort_key(self, item: Dict[str, Any], idx: int, sort_by: str):
        cost = item.get("cost") if item.get("cost") is not None else float("inf")
        rating = item.get("rating") if item.get("rating") is not None else 0.0
        if sort_by == "cost":
            return (cost, -rating, idx)
        return (-rating, cost, idx)

    def query_restaurants(self, slot, filt: Dict[str, Any], state: Any, cap: int = 80) -> List[Dict]:
        city = self._slot_attr(slot, "city") or self._slot_attr(slot, "destination")
        if not city:
            return []
        city_norm = self._normalize_city(city)
        restaurants = list(self._restaurant_buckets.get(city_norm, []))
        avoid_ids = {str(x) for x in (filt.get("avoid_ids") or [])}
        avoid_ids |= self._used_restaurant_ids(state)

        cuisines = [str(c).lower() for c in (filt.get("cuisines") or []) if c]
        max_cost = filt.get("max_cost")
        min_rating = filt.get("min_rating")

        df = pd.DataFrame(restaurants)
        if not df.empty:
            df = df[~df["id"].astype(str).isin(avoid_ids)]
            if cuisines:
                df = df[df["cuisines_lc"].apply(lambda s: any(c in (s or "") for c in cuisines))]
            df = self._apply_numeric_cap(df, "cost", max_cost)
            if min_rating is not None and "rating" in df.columns:
                try:
                    df = df[df["rating"].notna() & (df["rating"].astype(float) >= float(min_rating))]
                except Exception:
                    pass

        filtered: List[Tuple[Dict, int]] = []
        for idx, row in df.reset_index(drop=True).iterrows() if not df.empty else []:
            filtered.append((row.to_dict(), idx))

        if not filtered:
            return []
        sort_by = str(filt.get("sort_by") or "rating").lower()
        filtered.sort(key=lambda pair: self._restaurant_sort_key(pair[0], pair[1], sort_by))
        candidates = [item for item, _ in filtered[:cap]]
        if max_cost is not None:
            try:
                mc = float(max_cost)
                bad = [c for c in candidates if c.get("cost") is not None and float(c["cost"]) > mc + 1e-6]
                if bad:
                    raise RuntimeError(
                        f"[KB] meal cap violated max_cost={mc} bad0_cost={bad[0].get('cost')} bad0={bad[0]}"
                    )
            except Exception:
                pass
        return candidates

    def query_attractions(self, slot, filt: Dict[str, Any], state: Any, cap: int = 80) -> List[Dict]:
        city = self._slot_attr(slot, "city") or self._slot_attr(slot, "destination")
        if not city:
            return []
        city_norm = self._normalize_city(city)
        attractions = list(self._attraction_buckets.get(city_norm, []))
        avoid_ids = {str(x) for x in (filt.get("avoid_ids") or [])}
        avoid_ids |= self._used_attraction_ids(state)

        df = pd.DataFrame(attractions)
        if not df.empty:
            df = df[~df["id"].astype(str).isin(avoid_ids)]
            df = self._apply_numeric_cap(df, "cost", filt.get("max_cost"))
        filtered: List[Tuple[Dict, int]] = []
        for idx, row in df.reset_index(drop=True).iterrows() if not df.empty else []:
            filtered.append((row.to_dict(), idx))

        if not filtered:
            return []
        sort_by = str(filt.get("sort_by") or "rating").lower()
        if sort_by == "name":
            filtered.sort(key=lambda pair: (pair[0].get("name") or "", pair[1]))
        elif sort_by == "distance":
            filtered.sort(key=lambda pair: pair[1])
        else:
            filtered.sort(
                key=lambda pair: (
                    -(pair[0].get("rating") if pair[0].get("rating") is not None else 0.0),
                    pair[1],
                )
            )
        candidates = [item for item, _ in filtered[:cap]]
        max_cost = filt.get("max_cost")
        if max_cost is not None:
            try:
                mc = float(max_cost)
                bad = [c for c in candidates if c.get("cost") is not None and float(c["cost"]) > mc + 1e-6]
                if bad:
                    raise RuntimeError(
                        f"[KB] attraction cap violated max_cost={mc} bad0_cost={bad[0].get('cost')} bad0={bad[0]}"
                    )
            except Exception:
                pass
        return candidates

    def fallback_candidates(self, slot, state: Any, cap: int = 20) -> List[Any]:
        stype = self._slot_attr(slot, "type")
        stype = str(stype or "").lower()
        if stype in ("segment", "flight"):
            origin = self._slot_attr(slot, "origin") or self._slot_attr(slot, "from")
            destination = self._slot_attr(slot, "destination") or self._slot_attr(slot, "to")
            if not origin or not destination:
                return []
            pair = (self._normalize_city(origin), self._normalize_city(destination))
            return list(self._flight_buckets.get(pair, []))[:cap]
        if stype == "hotel":
            city = self._slot_attr(slot, "city") or self._slot_attr(slot, "destination")
            if not city:
                return []
            return list(self._accommodation_buckets.get(self._normalize_city(city), []))[:cap]
        if stype == "restaurant":
            city = self._slot_attr(slot, "city") or self._slot_attr(slot, "destination")
            if not city:
                return []
            return list(self._restaurant_buckets.get(self._normalize_city(city), []))[:cap]
        if stype == "attraction":
            city = self._slot_attr(slot, "city") or self._slot_attr(slot, "destination")
            if not city:
                return []
            return list(self._attraction_buckets.get(self._normalize_city(city), []))[:cap]
        return []
