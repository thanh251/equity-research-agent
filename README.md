# Equity Research Agent

AI agent so sánh 2–5 cổ phiếu Mỹ trong một yêu cầu. Ứng dụng hiển thị giá và nguồn dữ liệu thị trường, đối chiếu các chỉ tiêu tài chính và tạo báo cáo tiếng Việt có bằng chứng từ hồ sơ SEC.

Kiến trúc kế thừa từ [ai-agent-moat của Erik Taylor](https://github.com/eriktaylor/ai-agent-moat). Workflow mới xử lý nhiều mã, tính các tỷ lệ bằng Python và giữ thông tin nguồn theo từng chỉ tiêu.

## Chạy ứng dụng

Cần Python 3.12 và [uv](https://docs.astral.sh/uv/).

```bash
uv sync --python 3.12
cp .env.example .env
uv run streamlit run app.py
```

Điền thông tin trong `.env` hoặc dùng thanh bên của ứng dụng.

- `SEC_USER_AGENT` gồm tên dự án và email liên hệ, ví dụ `Equity Research contact@example.com`.
- `APCA_API_KEY_ID` và `APCA_API_SECRET_KEY` bật Alpaca. Feed mặc định là `iex`, lấy dữ liệu từ sàn IEX. `sip` cần quyền truy cập tương ứng. Khi thiếu snapshot, bảng hiển thị giá Yahoo Finance nếu có.
- `ORCAROUTER_API_KEY` bật báo cáo DeepSeek. `ORCAROUTER_MODEL` chọn model ID theo tài khoản. Bảng so sánh vẫn chạy khi không có khóa này. Lần tạo báo cáo đầu tiên tải embedding model từ Hugging Face.

## Demo

Nhập `NVDA, AMD` và chọn tiêu chí tăng trưởng, biên lợi nhuận, định giá, rủi ro cạnh tranh hoặc mảng chip AI. Mở phần nguồn theo chỉ tiêu để xem kỳ dữ liệu, sau đó đối chiếu nhận định AI với đoạn trích SEC tương ứng.

## Luồng dữ liệu

1. Alpaca REST lấy snapshot giá, bid/ask và thời điểm quan sát. Yahoo Finance bổ sung vốn hóa và P/E.
2. SEC Company Facts cung cấp dữ liệu năm tài chính. Python tính tăng trưởng so với kỳ năm trước và biên lợi nhuận hoạt động, ghi rõ ngày bắt đầu và kết thúc kỳ.
3. Hồ sơ 10-K và 10-Q gần nhất của từng mã được lập chỉ mục FAISS trong bộ nhớ với `all-MiniLM-L6-v2`. Agent truy xuất đoạn trích theo mã và tiêu chí.
4. DeepSeek qua Orca Router viết bản tổng hợp. Pydantic kiểm tra cấu trúc, sau đó ứng dụng xác thực ID bằng chứng có tồn tại trong dữ liệu đã truy xuất.

Bảng tài chính dùng năm tài chính gần nhất của từng công ty, nên ngày kết thúc kỳ giữa các mã có thể khác nhau. Dữ liệu thiếu hiển thị bằng dấu gạch ngang. Giá và P/E có thể đến từ các nguồn và thời điểm khác nhau.

## Kiểm tra

```bash
uv run python -m unittest discover -s tests
```
