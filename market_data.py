"""Market snapshots and company filings for US equities."""

import re
from dataclasses import dataclass
from datetime import date

import httpx
import yfinance as yf
from bs4 import BeautifulSoup

SEC_DATA = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)


@dataclass(frozen=True)
class AnnualFact:
    start: date
    end: date
    filed: date
    value: float
    accession: str


@dataclass(frozen=True)
class Filing:
    symbol: str
    form: str
    period: str
    filed: str
    url: str


def parse_symbols(value):
    symbols = list(dict.fromkeys(part.upper() for part in re.split(r"[,\s]+", value.strip()) if part))
    if not 2 <= len(symbols) <= 5 or any(
        not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", symbol) for symbol in symbols
    ):
        raise ValueError("Nhập từ 2 đến 5 mã cổ phiếu Mỹ khác nhau, ví dụ NVDA, AMD.")
    return symbols


def annual_facts(concepts, tag):
    by_end = {}
    for row in concepts.get(tag, {}).get("units", {}).get("USD", []):
        if row.get("form") not in {"10-K", "10-K/A"} or not row.get("start"):
            continue
        start = date.fromisoformat(row["start"])
        end = date.fromisoformat(row["end"])
        if not 350 <= (end - start).days <= 380:
            continue
        fact = AnnualFact(start, end, date.fromisoformat(row["filed"]), row["val"], row["accn"])
        previous = by_end.get(end)
        if previous is None or (fact.filed, fact.accession) > (previous.filed, previous.accession):
            by_end[end] = fact
    return by_end


def annual_comparison(concepts):
    revenue = {}
    for tag in REVENUE_TAGS:
        for period, fact in annual_facts(concepts, tag).items():
            revenue.setdefault(period, fact)
    operating_income = annual_facts(concepts, "OperatingIncomeLoss")
    if not revenue:
        return None, None, None
    latest = revenue[max(revenue)]
    previous = next(
        (revenue[end] for end in sorted(revenue, reverse=True) if 350 <= (latest.end - end).days <= 380),
        None,
    )
    income = operating_income.get(latest.end)
    if income and income.start != latest.start:
        income = None
    return latest, previous, income


def sec_symbols(client):
    response = client.get("https://www.sec.gov/files/company_tickers.json")
    response.raise_for_status()
    return {row["ticker"].upper(): row["cik_str"] for row in response.json().values()}


def filing_directory(cik, accession):
    return f"{SEC_ARCHIVES}/{cik}/{accession.replace('-', '')}"


def company_financials(client, cik):
    response = client.get(f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json")
    response.raise_for_status()
    return annual_comparison(response.json()["facts"].get("us-gaap", {}))


def recent_filings(client, symbol, cik):
    response = client.get(f"{SEC_DATA}/submissions/CIK{cik:010d}.json")
    response.raise_for_status()
    rows = response.json()["filings"]["recent"]
    filings = {}
    for form, period, filed, accession, document in zip(
        rows["form"], rows["reportDate"], rows["filingDate"],
        rows["accessionNumber"], rows["primaryDocument"],
    ):
        if form in {"10-K", "10-Q"} and form not in filings and document:
            filings[form] = Filing(
                symbol, form, period, filed, f"{filing_directory(cik, accession)}/{document}",
            )
        if len(filings) == 2:
            break
    return list(filings.values())


def filing_text(client, filing):
    response = client.get(filing.url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup.find_all(["script", "style", "ix:header", "table"]):
        tag.decompose()
    return re.sub(r"\n{2,}", "\n", soup.get_text("\n", strip=True))


def alpaca_snapshots(client, symbols, key, secret, feed):
    response = client.get(
        "https://data.alpaca.markets/v2/stocks/snapshots",
        params={"symbols": ",".join(symbols), "feed": feed, "currency": "USD"},
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
    )
    response.raise_for_status()
    return response.json()


def yahoo_metrics(symbol):
    data = yf.Ticker(symbol).get_info()
    return {key: data.get(key) for key in (
        "marketCap", "trailingPE", "regularMarketPrice", "regularMarketTime", "currency",
    )}
