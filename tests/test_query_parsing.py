from __future__ import annotations

import unittest

from mcts.travel.semantic.query_parsing import fallback_parse


class TestQueryParsing(unittest.TestCase):
    def test_fallback_parse_starting_covering_state(self):
        q = (
            "Can you assist in creating a 5-day travel itinerary starting in Sacramento and "
            "covering 2 cities in Washington state from March 22nd to March 26th, 2022?"
        )
        parsed = fallback_parse(q)
        self.assertEqual(parsed.get("origin"), "Sacramento")
        self.assertEqual(parsed.get("destination"), "Washington state")


if __name__ == "__main__":
    unittest.main()

