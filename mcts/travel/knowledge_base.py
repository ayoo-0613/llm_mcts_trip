from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import pandas as pd


@dataclass
class TripGoal:
    origin: str
    destination: str
    start_date: Optional[str] = None
    duration_days: Optional[int] = None
    return_required: bool = True
    budget: Optional[float] = None
    require_flight: bool = True
    require_accommodation: bool = True
    visiting_city_number: int = 1
    num_restaurants: int = 1
    num_attractions: int = 1
    meals_per_day: int = 3
    attractions_per_day_min: int = 2
    attractions_per_day_max: int = 3
    preferences: List[str] = field(default_factory=list)
    must_visit_cities: List[str] = field(default_factory=list)
    priority_cities: List[str] = field(default_factory=list)
    candidate_cities: List[str] = field(default_factory=list)
    fixed_city_order: List[str] = field(default_factory=list)
    transport_allowed_modes: Optional[List[str]] = None
    transport_forbidden_modes: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def as_text(self) -> str:
        prefs = ", ".join(self.preferences) if self.preferences else "None"
        budget_text = f"${self.budget:.0f}" if self.budget is not None else "unspecified"
        days_text = f"{self.duration_days} day(s)" if self.duration_days else "unspecified duration"
        cities_txt = (
            " | ".join(self.fixed_city_order)
            if self.fixed_city_order
            else (" | ".join(self.must_visit_cities) if self.must_visit_cities else "flexible")
        )
        parts = [
            f"Trip from {self.origin} to {self.destination}",
            f"start date: {self.start_date or 'unspecified'}",
            f"duration: {days_text}",
            f"budget: {budget_text}",
            f"preferences: {prefs}",
            f"city count target: {self.visiting_city_number}",
            f"city order: {cities_txt}",
        ]
        if self.notes:
            parts.append(f"notes: {self.notes}")
        return "; ".join(parts)


class TravelKnowledgeBase:
    def __init__(self, root: str = "database"):
        self.root = root
        self.flights = self._load_csv("flights/clean_Flights_2022.csv")
        self.accommodations = self._load_csv("accommodations/clean_accommodations_2022.csv")
        self.restaurants = self._load_csv("restaurants/clean_restaurant_2022.csv")
        self.attractions = self._load_csv("attractions/attractions.csv")
        self.distances = self._load_csv("googleDistanceMatrix/distance.csv")

        self._normalize()

    def _load_csv(self, relative_path: str) -> pd.DataFrame:
        path = os.path.join(self.root, relative_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected dataset at {path}")
        return pd.read_csv(path)

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

        self.attractions["City_norm"] = self.attractions["City"].apply(self._normalize_city)

        if "origin" in self.distances:
            self.distances["origin_norm"] = self.distances["origin"].apply(self._normalize_city)
            self.distances["destination_norm"] = self.distances["destination"].apply(self._normalize_city)

    def get_flights(self, origin: str, destination: str, top_k: int = 5,
                    max_price: Optional[float] = None) -> List[Dict]:
        orig = self._normalize_city(origin)
        dest = self._normalize_city(destination)
        df = self.flights[
            (self.flights["OriginCityName_norm"] == orig) &
            (self.flights["DestCityName_norm"] == dest)
        ]
        if max_price is not None:
            df = df[df["Price"] <= max_price]
        df = df.sort_values(by=["Price", "ActualElapsedTime"], ascending=[True, True]).head(top_k)
        return [
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
            for _, row in df.iterrows()
        ]

    def get_accommodations(self, city: str, top_k: int = 5,
                           max_price: Optional[float] = None) -> List[Dict]:
        city_norm = self._normalize_city(city)
        df = self.accommodations[self.accommodations["city_norm"] == city_norm]
        if max_price is not None:
            df = df[df["price"] <= max_price]
        df = df.sort_values(by=["price", "review rate number"], ascending=[True, False]).head(top_k)
        return [
            {
                "id": str(idx),
                "name": row["NAME"],
                "price": float(row["price"]),
                "room_type": row["room type"],
                "review": row.get("review rate number"),
                "occupancy": row.get("maximum occupancy"),
                "city": row["city"],
                "house_rules": row.get("house_rules"),
            }
            for idx, row in df.iterrows()
        ]

    def get_restaurants(self, city: str, preferences: Optional[List[str]] = None,
                        top_k: int = 5) -> List[Dict]:
        city_norm = self._normalize_city(city)
        df = self.restaurants[self.restaurants["City_norm"] == city_norm]
        prefs = [self._normalize_city(p) for p in preferences] if preferences else []
        if prefs:
            df = df[df["Cuisines"].apply(
                lambda c: any(pref in str(c).lower() for pref in prefs)
            )]
        df = df.sort_values(by=["Aggregate Rating", "Average Cost"], ascending=[False, True]).head(top_k)
        return [
            {
                "id": str(idx),
                "name": row["Name"],
                "city": row["City"],
                "cuisines": row["Cuisines"],
                "cost": float(row["Average Cost"]),
                "rating": float(row["Aggregate Rating"]),
            }
            for idx, row in df.iterrows()
        ]

    def get_attractions(self, city: str, top_k: int = 5) -> List[Dict]:
        city_norm = self._normalize_city(city)
        df = self.attractions[self.attractions["City_norm"] == city_norm]
        df = df.head(top_k)
        return [
            {
                "id": str(idx),
                "name": row["Name"],
                "address": row["Address"],
                "phone": row.get("Phone"),
                "website": row.get("Website"),
                "city": row["City"],
            }
            for idx, row in df.iterrows()
        ]

    def get_candidate_cities(self, destination_hint: Optional[str] = None,
                             must_visit: Optional[List[str]] = None,
                             priority: Optional[List[str]] = None,
                             top_k: int = 10) -> List[str]:
        must_visit = must_visit or []
        priority = priority or []
        dest_norm = self._normalize_city(destination_hint) if destination_hint else None

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
            match = re.search(r"([0-9]+(?:\\.[0-9]+)?)", str(raw))
            if not match:
                return None
            return float(match.group(1))
        except Exception:
            return None
