"""Streamlit interface for the US equity comparison workflow."""

import os
from datetime import datetime, timezone
from math import isfinite

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from market_data import (
    alpaca_snapshots,
    company_financials,
    filing_directory,
    filing_text,
    parse_symbols,
    recent_filings,
    sec_symbols,
    yahoo_metrics,
)
from research import retrieve_evidence, write_report

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash-1017"
CRITERIA = {
    "Tăng trưởng doanh thu": "annual revenue growth drivers and segment performance",
    "Biên lợi nhuận hoạt động": "operating margin changes and cost or product mix drivers",
    "Định giá": "business outlook and factors affecting expected earnings",
    "Rủi ro cạnh tranh": "competition and material risk factors",
    "Mảng chip AI": "AI accelerators, data center GPUs, segment disclosures and competition",
}


@st.cache_resource(show_spinner=False)
def embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def timestamp(value):
    if isinstance(value, (float, int)) and isfinite(value):
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    return value


def run_analysis(symbols, criteria, user_agent, alpaca_key, alpaca_secret, feed, orca_key, model):
    rows, filings, documents, warnings = [], [], [], []

    def add(symbol, metric, value, unit, period, provider, url):
        rows.append({
            "Mã": symbol, "Chỉ tiêu": metric, "Giá trị": value,
            "Đơn vị": unit, "Kỳ / thời điểm": period, "Nguồn": provider, "URL": url,
        })

    with httpx.Client(timeout=20, headers={"User-Agent": user_agent}, follow_redirects=True) as sec:
        try:
            ciks = sec_symbols(sec)
        except (httpx.HTTPError, ValueError):
            warnings.append("Không tải được danh sách mã từ SEC EDGAR.")
            ciks = {}

        snapshots = {}
        if alpaca_key and alpaca_secret:
            try:
                with httpx.Client(timeout=15) as alpaca:
                    snapshots = alpaca_snapshots(alpaca, symbols, alpaca_key, alpaca_secret, feed)
            except httpx.HTTPError:
                warnings.append("Không lấy được snapshot Alpaca. Kiểm tra khóa và quyền truy cập feed.")
        else:
            warnings.append("Chưa cấu hình Alpaca. Giá từ Yahoo Finance sẽ được hiển thị nếu có.")

        for symbol in symbols:
            try:
                yahoo = yahoo_metrics(symbol)
            except Exception:
                yahoo = {}
                warnings.append(f"{symbol}: Yahoo Finance không trả được dữ liệu thị trường.")

            trade = (snapshots.get(symbol) or {}).get("latestTrade") or {}
            quote = (snapshots.get(symbol) or {}).get("latestQuote") or {}
            yahoo_url = f"https://finance.yahoo.com/quote/{symbol}/"
            alpaca_url = "https://docs.alpaca.markets/us/reference/stocksnapshots-1"
            yahoo_time = timestamp(yahoo.get("regularMarketTime"))

            if trade.get("p") is not None:
                add(symbol, "Giá giao dịch gần nhất", trade["p"], "USD", trade.get("t"),
                    f"Alpaca {feed.upper()}", alpaca_url)
            else:
                add(symbol, "Giá giao dịch gần nhất", yahoo.get("regularMarketPrice"),
                    yahoo.get("currency"), yahoo_time, "Yahoo Finance", yahoo_url)
            if quote:
                add(symbol, "Giá mua (bid)", quote.get("bp"), "USD", quote.get("t"),
                    f"Alpaca {feed.upper()}", alpaca_url)
                add(symbol, "Giá bán (ask)", quote.get("ap"), "USD", quote.get("t"),
                    f"Alpaca {feed.upper()}", alpaca_url)
            add(symbol, "Vốn hóa", yahoo.get("marketCap"), yahoo.get("currency"), None,
                "Yahoo Finance", yahoo_url)
            add(symbol, "P/E quá khứ", yahoo.get("trailingPE"), "x", None, "Yahoo Finance", yahoo_url)

            cik = ciks.get(symbol)
            revenue = previous = income = None
            facts_url = None
            if cik:
                facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
                try:
                    revenue, previous, income = company_financials(sec, cik)
                except (httpx.HTTPError, ValueError, KeyError):
                    warnings.append(f"{symbol}: Không lấy được số liệu tài chính từ SEC.")
                try:
                    filings.extend(recent_filings(sec, symbol, cik))
                except (httpx.HTTPError, ValueError, KeyError):
                    warnings.append(f"{symbol}: Không lấy được danh sách hồ sơ SEC.")
            else:
                warnings.append(f"{symbol}: Không tìm thấy mã trong danh sách SEC đã tải.")

            period = f"{revenue.start} đến {revenue.end}" if revenue else None
            source = filing_directory(cik, revenue.accession) if revenue else facts_url
            add(symbol, "Doanh thu năm tài chính", revenue.value if revenue else None,
                "USD", period, "SEC EDGAR", source)
            growth = (revenue.value / previous.value - 1) * 100 if revenue and previous and previous.value else None
            growth_period = f"{period} so với kỳ kết thúc {previous.end}" if growth is not None else period
            add(symbol, "Tăng trưởng doanh thu năm", growth, "%", growth_period, "SEC EDGAR", facts_url)
            margin = 100 * income.value / revenue.value if income and revenue.value else None
            add(symbol, "Biên lợi nhuận hoạt động năm", margin, "%", period, "SEC EDGAR", facts_url)

        if orca_key:
            for filing in filings:
                try:
                    documents.append((filing, filing_text(sec, filing)))
                except httpx.HTTPError:
                    warnings.append(f"{filing.symbol}: Không tải được nội dung {filing.form}.")

    evidence, report = {}, None
    if not orca_key:
        warnings.append("Chưa cấu hình Orca Router. Bảng số liệu và danh sách hồ sơ vẫn dùng được.")
    elif documents:
        try:
            evidence = retrieve_evidence(embedding_model(), documents, [CRITERIA[item] for item in criteria])
            if not evidence:
                warnings.append("Không tìm thấy đoạn tài liệu phù hợp để viết báo cáo.")
            else:
                metrics = [
                    {**row, "Giá trị": None if pd.isna(row["Giá trị"]) else row["Giá trị"]}
                    for row in rows
                ]
                report = write_report(orca_key, model, symbols, criteria, metrics, evidence)
        except Exception as error:
            status = f" HTTP {error.response.status_code}." if isinstance(error, httpx.HTTPStatusError) else ""
            warnings.append(f"Không tạo được báo cáo AI.{status} Kiểm tra cấu hình model và kết nối.")
    else:
        warnings.append("Chưa có nội dung hồ sơ SEC để tạo báo cáo có dẫn nguồn.")
    return rows, filings, evidence, report, warnings


def render_result(result):
    rows, filings, evidence, report, warnings = result
    for warning in warnings:
        st.warning(warning)

    frame = pd.DataFrame(rows)
    table = frame.copy()
    table["Giá trị"] = table["Giá trị"].map(lambda value: f"{value:,.2f}" if pd.notna(value) else "—")
    table["Hiển thị"] = table["Giá trị"] + " " + table["Đơn vị"].fillna("")
    comparison = table.pivot(index="Chỉ tiêu", columns="Mã", values="Hiển thị")
    st.subheader("Bảng so sánh")
    st.dataframe(comparison.reindex(columns=frame["Mã"].unique()), width="stretch")
    st.caption("Các chỉ tiêu tài chính dùng năm tài chính gần nhất có dữ liệu của từng công ty.")
    with st.expander("Kỳ dữ liệu và nguồn theo chỉ tiêu"):
        st.dataframe(frame, hide_index=True, width="stretch", column_config={"URL": st.column_config.LinkColumn("Nguồn gốc")})
    st.download_button("Tải bảng CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name="stock-comparison.csv", mime="text/csv")

    if report:
        st.subheader("Báo cáo so sánh")
        st.text(report.summary)
        for finding in report.findings:
            st.text(finding.claim)
            st.caption("Bằng chứng: " + ", ".join(finding.evidence_ids))
        if report.limitations:
            st.markdown("**Giới hạn của phân tích**")
            for limitation in report.limitations:
                st.text(limitation)
    if evidence:
        st.subheader("Bằng chứng truy xuất")
        for source_id, source in evidence.items():
            with st.expander(f"{source_id} · {source['symbol']} · {source['form']} · {source['period']}"):
                st.link_button("Mở hồ sơ SEC", source["url"])
                st.caption(f"Ngày nộp: {source['filed']}")
                st.text(source["excerpt"])
    if filings:
        with st.expander("Hồ sơ SEC đã tìm thấy"):
            for filing in filings:
                st.link_button(f"{filing.symbol} · {filing.form} · {filing.period}", filing.url)


def main():
    load_dotenv()
    st.set_page_config(page_title="So sánh cổ phiếu Mỹ", layout="wide")
    st.title("AI Agent nghiên cứu cổ phiếu Mỹ")
    st.write("So sánh nhiều mã theo cùng chỉ tiêu, với dữ liệu thị trường và hồ sơ SEC.")
    with st.sidebar:
        user_agent = st.text_input("SEC User-Agent", value=os.getenv("SEC_USER_AGENT", ""))
        alpaca_key = st.text_input("Alpaca API Key", value=os.getenv("APCA_API_KEY_ID", ""), type="password")
        alpaca_secret = st.text_input("Alpaca Secret Key", value=os.getenv("APCA_API_SECRET_KEY", ""), type="password")
        feeds = ["iex", "sip", "delayed_sip"]
        configured_feed = os.getenv("ALPACA_FEED", "iex")
        feed = st.selectbox("Alpaca feed", feeds, index=feeds.index(configured_feed) if configured_feed in feeds else 0)
        orca_key = st.text_input("Orca Router API Key", value=os.getenv("ORCAROUTER_API_KEY", ""), type="password")
        model = st.text_input("Orca Router model", value=os.getenv("ORCAROUTER_MODEL") or DEFAULT_MODEL)

    with st.form("comparison"):
        raw_symbols = st.text_input("Mã cổ phiếu", "NVDA, AMD")
        criteria = st.multiselect("Tiêu chí phân tích", list(CRITERIA), default=list(CRITERIA))
        submitted = st.form_submit_button("Phân tích", type="primary")
    if submitted:
        st.session_state.pop("result", None)
        try:
            symbols = parse_symbols(raw_symbols)
            if "@" not in user_agent:
                raise ValueError("Nhập SEC User-Agent gồm tên dự án và email liên hệ.")
            if not criteria:
                raise ValueError("Chọn ít nhất một tiêu chí phân tích.")
        except ValueError as error:
            st.error(str(error))
        else:
            with st.spinner("Đang thu thập và đối chiếu dữ liệu…"):
                st.session_state["result"] = run_analysis(
                    symbols, criteria, user_agent, alpaca_key, alpaca_secret, feed, orca_key, model,
                )
    if "result" in st.session_state:
        render_result(st.session_state["result"])


if __name__ == "__main__":
    main()
