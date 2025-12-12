from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
import os
import pandas as pd


@dataclass
class TripGoal:
    # === 核心来自 JSON ===
    origin: str                    # JSON: org
    destination: str               # JSON: dest
    start_date: Optional[str]      # JSON: date[0]（如果有）
    duration_days: int             # JSON: days
    visiting_city_number: int      # JSON: visiting_city_number
    budget: Optional[float]        # JSON: budget
    people_number: int = 1         # JSON: people_number（可选）

    # local_constraint 拆成更结构化的字段（或直接保留原 dict 也可以）
    house_rule: Optional[List[str]] = None      # local_constraint["house rule"]
    cuisine: Optional[List[str]] = None         # local_constraint["cuisine"]
    room_type: Optional[List[str]] = None       # local_constraint["room type"]
    transportation: Optional[List[str]] = None  # local_constraint["transportation"]

    # === 内部规划用的默认约束（不是 JSON 里有，而是你系统需要）===
    return_required: bool = True
    require_flight: bool = True
    require_accommodation: bool = True

    meals_per_day: int = 3
    attractions_per_day_min: int = 2
    attractions_per_day_max: int = 3

    # 偏好（可从 cuisine / NL 里抽取）
    preferences: List[str] = field(default_factory=list)
    # 规划扩展
    must_visit_cities: List[str] = field(default_factory=list)
    priority_cities: List[str] = field(default_factory=list)
    candidate_cities: List[str] = field(default_factory=list)
    fixed_city_order: List[str] = field(default_factory=list)
    transport_allowed_modes: Optional[List[str]] = None
    transport_forbidden_modes: List[str] = field(default_factory=list)
    num_restaurants: int = 1
    num_attractions: int = 1
    notes: Optional[str] = None

    def as_text(self) -> str:
        prefs = ", ".join(self.preferences) if self.preferences else "None"
        budget_text = f"${self.budget:.0f}" if self.budget is not None else "unspecified"
        days_text = f"{self.duration_days} day(s)"
        parts = [
            f"Trip from {self.origin} to {self.destination}",
            f"start date: {self.start_date or 'unspecified'}",
            f"duration: {days_text}",
            f"budget: {budget_text}",
            f"people: {self.people_number}",
            f"city count target: {self.visiting_city_number}",
            f"preferences: {prefs}",
            f"must cities: {', '.join(self.must_visit_cities) or 'None'}",
            f"priority cities: {', '.join(self.priority_cities) or 'None'}",
        ]
        return "; ".join(parts)


    @staticmethod
    def trip_goal_from_json(data: Dict[str, Any]) -> "TripGoal":
        """把 NL 抽取出来的 JSON 转成 TripGoal 对象。"""
        lc = data.get("local_constraint") or {}
        dates = data.get("date") or []
        return TripGoal(
            origin=data["org"],
            destination=data["dest"],
            start_date=dates[0] if dates else None,
            duration_days=data["days"],
            visiting_city_number=data.get("visiting_city_number", 1),
            budget=data.get("budget"),
            people_number=data.get("people_number", 1),
            house_rule=lc.get("house rule"),
            cuisine=lc.get("cuisine"),
            room_type=lc.get("room type"),
            transportation=lc.get("transportation"),
            must_visit_cities=data.get("must_visit_cities", []),
            priority_cities=data.get("priority_cities", []),
            candidate_cities=data.get("candidate_cities", []),
            fixed_city_order=data.get("fixed_city_order") or [],
            transport_allowed_modes=data.get("transport_allow") or data.get("transport_allowed_modes"),
            transport_forbidden_modes=data.get("transport_forbid") or data.get("transport_forbidden_modes", []),
            num_restaurants=data.get("restaurants", 1),
            num_attractions=data.get("attractions", 1),
            notes=data.get("notes"),
        )


class TravelKnowledgeBase:
    def __init__(self, root: str = "database"):
        self.root = root
        self.flights = self._load_csv("flights/clean_Flights_2022.csv")
        self.accommodations = self._load_csv("accommodations/clean_accommodations_2022.csv")
        self.restaurants = self._load_csv("restaurants/clean_restaurant_2022.csv")
        self.attractions = self._load_csv("attractions/attractions.csv")
        self.distances = self._load_csv("googleDistanceMatrix/distance.csv")
        self.states, self.cities, self.city_to_state, self.state_to_cities = self._load_background()

        self._normalize()
        self._build_indexes()

    def _load_csv(self, relative_path: str) -> pd.DataFrame:
        path = os.path.join(self.root, relative_path)
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
            stay = {
                "id": str(idx),
                "name": row["NAME"],
                "price": float(row["price"]) if not pd.isna(row["price"]) else None,
                "room_type": row["room type"],
                "review": row.get("review rate number"),
                "occupancy": row.get("maximum occupancy"),
                "city": row["city"],
                "house_rules": row.get("house_rules"),
            }
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
                           max_price: Optional[float] = None) -> List[Dict]:
        city_norm = self._normalize_city(city)
        stays = self._accommodation_buckets.get(city_norm, [])
        if max_price is not None:
            stays = [s for s in stays if s.get("price") is not None and s["price"] <= max_price]
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
