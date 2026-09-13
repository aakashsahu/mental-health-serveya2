# Mental Health in Tech — EDA Dashboard

An interactive Streamlit rebuild of the `eda_mental_health.ipynb` notebook, using
the 2014 OSMI Mental Health in Tech Survey (`survey.csv`, 1,259 responses).

## Files
- `app.py` — the Streamlit app
- `survey.csv` — the dataset (must sit next to `app.py`)
- `requirements.txt` — pinned dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL Streamlit prints (usually http://localhost:8501).

## Deploy for free — Streamlit Community Cloud
1. Create a new GitHub repo and push these three files (`app.py`, `survey.csv`,
   `requirements.txt`) to it.
2. Go to https://share.streamlit.io, sign in with GitHub, click **"New app"**.
3. Pick your repo/branch, set **Main file path** to `app.py`, click **Deploy**.
4. Your app gets a public URL like `https://<your-app>.streamlit.app`.

No server config needed — Streamlit Cloud installs `requirements.txt`
automatically and restarts the app on every push to the branch.

## Deploy elsewhere
- **Hugging Face Spaces**: create a Space with SDK = Streamlit, upload the same
  three files (name the entry file `app.py`).
- **Docker / any host**: use the snippet below.
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY . .
  RUN pip install --no-cache-dir -r requirements.txt
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```
- **Render / Railway / Fly.io**: point the start command at
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## What's in the app
- Sidebar filters: gender, age range, country, tech-company status — every
  chart updates live.
- **Overview** — headline metrics + missing-value summary from the raw file.
- **Demographics** — age, gender, country, company-size distributions.
- **Treatment & Work** — treatment rates by gender/age/remote-work/family
  history, work-interference levels.
- **Correlations** — full encoded correlation heatmap + top correlates of
  `treatment`.
- **Attitudes & Support** — willingness to disclose to coworkers/supervisors,
  interview disclosure, employer benefit/wellness-program coverage, tech vs
  non-tech comparison.
- **Geography** — treatment rate by top-10 countries, top 15 US states.
- **Key Findings** — the notebook's summary bullets computed live from data,
  plus a 4-panel recap dashboard.

Cleaning logic (age clipping/imputation, gender standardization, NA handling)
mirrors the source notebook exactly.
