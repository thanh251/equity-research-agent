import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.embeddings import FakeEmbeddings
from streamlit.testing.v1 import AppTest

from market_data import Filing, alpaca_snapshots, annual_comparison, parse_symbols
from research import retrieve_evidence, write_report

APP = Path(__file__).resolve().parents[1] / "app.py"


def fact(start, end, value, filed, form="10-K"):
    return {"start": start, "end": end, "val": value, "filed": filed, "form": form,
            "accn": "0000000001-25-000001"}


class WorkflowTests(unittest.TestCase):
    def test_symbols_deduplicate_and_reject_invalid_input(self):
        self.assertEqual(parse_symbols("nvda, AMD nvda"), ["NVDA", "AMD"])
        for invalid in ("NVDA", "NVDA,A<script>", "A,B,C,D,E,F"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_symbols(invalid)

    def test_annual_facts_select_comparable_periods(self):
        start, end = "2024-01-29", "2025-01-26"
        concepts = {
            "Revenues": {"units": {"USD": [
                fact("2023-01-30", "2024-01-28", 100, "2024-02-20"),
                fact(start, end, 200, "2025-02-25"),
                fact(start, end, 210, "2025-03-01", "10-K/A"),
                fact("2025-01-27", "2025-04-27", 99, "2025-05-27", "10-Q"),
            ]}},
            "OperatingIncomeLoss": {"units": {"USD": [
                fact("2024-02-01", end, 80, "2025-02-25"),
            ]}},
        }
        current, prior, income = annual_comparison(concepts)
        self.assertEqual((current.value, prior.value), (210, 100))
        self.assertIsNone(income)

    def test_alpaca_fetches_both_symbols_with_explicit_feed(self):
        def respond(request):
            self.assertEqual(request.url.params["symbols"], "NVDA,AMD")
            self.assertEqual(request.url.params["feed"], "iex")
            self.assertEqual(request.headers["APCA-API-KEY-ID"], "test-key")
            return httpx.Response(200, json={"NVDA": {"latestTrade": {"p": 130}}})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            result = alpaca_snapshots(client, ["NVDA", "AMD"], "test-key", "test-secret", "iex")
        self.assertEqual(result["NVDA"]["latestTrade"]["p"], 130)

    def test_report_checks_retrieved_source_ids(self):
        filings = [
            (Filing(symbol, "10-K", "2025-01-26", "2025-02-25", f"https://www.sec.gov/{symbol}"),
             f"{symbol} disclosed risks related to its supply chain.")
            for symbol in ("NVDA", "AMD")
        ]
        evidence = retrieve_evidence(FakeEmbeddings(size=8), filings, ["supply chain risks"])
        self.assertEqual({doc["symbol"] for doc in evidence.values()}, {"NVDA", "AMD"})
        payload = {"summary": "So sánh", "findings": [{"claim": "Rủi ro nguồn cung", "evidence_ids": ["E1"]}],
                   "limitations": []}
        with patch("research.httpx.post") as post:
            def response():
                return httpx.Response(
                    200, request=httpx.Request("POST", "https://api.orcarouter.ai/v1/chat/completions"),
                    json={"choices": [{"message": {"content": json.dumps(payload)}}]},
                )
            post.side_effect = lambda *args, **kwargs: response()
            args = ("test-key", "test-model", ["NVDA", "AMD"], ["Rủi ro"], [], evidence)
            self.assertEqual(write_report(*args).findings[0].evidence_ids, ["E1"])
            payload["findings"][0]["evidence_ids"] = ["E99"]
            with self.assertRaisesRegex(ValueError, "nguồn"):
                write_report(*args)

    def test_streamlit_nvda_amd_comparison(self):
        with (
            patch.dict(os.environ, {"ORCAROUTER_API_KEY": "", "APCA_API_KEY_ID": "", "APCA_API_SECRET_KEY": ""}),
            patch("market_data.sec_symbols", return_value={"NVDA": 1045810, "AMD": 2488}),
            patch("market_data.company_financials", return_value=(None, None, None)),
            patch("market_data.recent_filings", return_value=[]),
            patch("market_data.yahoo_metrics", return_value={"regularMarketPrice": 130, "currency": "USD"}),
        ):
            app = AppTest.from_file(str(APP)).run()
            next(field for field in app.text_input if field.label == "SEC User-Agent").set_value("Test test@example.com")
            app.button[0].click().run()
        self.assertFalse(app.exception)
        comparison = app.dataframe[0].value
        self.assertEqual(list(comparison.columns), ["NVDA", "AMD"])
        self.assertEqual(comparison.loc["Giá giao dịch gần nhất", "NVDA"], "130.00 USD")


if __name__ == "__main__":
    unittest.main()
