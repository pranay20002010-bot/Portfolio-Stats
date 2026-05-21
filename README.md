# Portfolio Stats (Streamlit)

Upload a transactions Excel file, view portfolio analytics, and download per-client reports (PDF/CSV).

## Expected Excel columns

Your `.xlsx` must include these columns (case-insensitive):

- `portfolio` (client name)
- `date`
- `ticker`
- `action` (`BUY` / `SELL`)
- `quantity`
- `price`

If the file is password-protected/encrypted, enter the password in the sidebar.

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (simple)

1. Put these files in the root of your GitHub repo:
   - `app.py`
   - `requirements.txt`
   - `portfolio_pipeline.py`
   - `portfolio_pdf.py`
   - `assets/vika_logo.png`
   - `.streamlit/config.toml` (optional theme)
2. In Streamlit Community Cloud, deploy from that GitHub repo and set the entrypoint to `app.py`.
3. When you change code, commit to the same branch you deployed (usually `main`) and reboot the app.
