# app.py
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Agentic ELT – Real-Time OpenAQ", layout="wide")
st.title("🧠 Agentic ELT Dashboard – Real-Time Air Quality (OpenAQ)")
st.caption("100% free & open-source pipeline. No API keys. Runs on Streamlit free tier.")

API_URL = "https://api.openaq.org/v2/latest"

# -----------------------------
# SESSION STATE (MEMORY)
# -----------------------------
if "etl_frame" not in st.session_state:
    st.session_state.etl_frame = 0
if "mem_insights" not in st.session_state:
    st.session_state.mem_insights = []   # rolling short-term memory of insights
if "prev_snapshot" not in st.session_state:
    st.session_state.prev_snapshot = None  # previous dataframe for deltas
if "last_run_meta" not in st.session_state:
    st.session_state.last_run_meta = {}

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("🔧 Controls")
country = st.sidebar.text_input("Country Code (e.g., IN, US, GB)", "IN").upper().strip()
limit = st.sidebar.slider("Number of Cities to Pull", 5, 50, 20)
ttl = st.sidebar.slider("Cache TTL (seconds)", 15, 180, 45)
show_logs = st.sidebar.checkbox("Show Agent Logs", value=False)

# -----------------------------
# UTILS
# -----------------------------
ETL_FRAMES = [
    "Extract 🔵 ➜ Transform ⬜ ➜ Load ⬜",
    "Extract 🔵🔵 ➜ Transform ⬜ ➜ Load ⬜",
    "Extract ⬜ ➜ Transform 🔵 ➜ Load ⬜",
    "Extract ⬜ ➜ Transform 🔵🔵 ➜ Load ⬜",
    "Extract ⬜ ➜ Transform ⬜ ➜ Load 🔵",
    "Extract ⬜ ➜ Transform ⬜ ➜ Load 🔵🔵",
]
def animated_etl():
    f = ETL_FRAMES[st.session_state.etl_frame % len(ETL_FRAMES)]
    st.session_state.etl_frame += 1
    return f

def keep_memory(msg: str, max_items: int = 8):
    st.session_state.mem_insights.append(msg)
    if len(st.session_state.mem_insights) > max_items:
        st.session_state.mem_insights = st.session_state.mem_insights[-max_items:]

# -----------------------------
# EXTRACTOR
# -----------------------------
@st.cache_data(ttl=ttl, show_spinner=False)
def fetch_openaq(country_code: str, limit_rows: int) -> pd.DataFrame:
    try:
        r = requests.get(
            API_URL,
            params={
                "country": country_code,
                "limit": limit_rows,
                "sort": "desc",
                "order_by": "lastUpdated",
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("results", [])
        rows = []
        for res in data:
            city = res.get("city")
            for m in res.get("measurements", []):
                rows.append({
                    "City": city,
                    "Parameter": m.get("parameter"),
                    "Value": m.get("value"),
                    "Unit": m.get("unit"),
                    "Last Updated": m.get("lastUpdated"),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        st.session_state.last_run_meta["extract_error"] = str(e)
        return pd.DataFrame()

# -----------------------------
# PLANNER (decides what to do)
# -----------------------------
def planner(df: pd.DataFrame) -> dict:
    """
    Simple rule-based planning:
    - If many cities but few parameters -> aggregate by City.
    - If many parameters -> aggregate by Parameter too.
    - If we have a previous snapshot -> compute deltas.
    """
    plan = {"aggregate_city": False, "aggregate_param": False, "compute_deltas": False}
    if df.empty:
        return plan

    nunique_cities = df["City"].nunique()
    nunique_params = df["Parameter"].nunique()

    plan["aggregate_city"] = nunique_cities >= 2
    plan["aggregate_param"] = nunique_params >= 2
    plan["compute_deltas"] = st.session_state.prev_snapshot is not None

    # store rationale for logs
    plan["_rationale"] = (
        f"Planner: cities={nunique_cities}, params={nunique_params}, "
        f"deltas={'yes' if plan['compute_deltas'] else 'no'}"
    )
    return plan

# -----------------------------
# TRANSFORMER (does the work)
# -----------------------------
def transformer(df: pd.DataFrame, plan: dict) -> dict:
    out = {"df": df, "avg_by_city": None, "avg_by_param": None, "deltas": None, "alerts": []}
    if df.empty:
        return out

    # Clean numeric
    df = df.copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    if plan.get("aggregate_city"):
        out["avg_by_city"] = (
            df.groupby("City", as_index=True)["Value"].mean().sort_values(ascending=False)
        )

    if plan.get("aggregate_param"):
        out["avg_by_param"] = (
            df.groupby("Parameter", as_index=True)["Value"].mean().sort_values(ascending=False)
        )

    # Simple z-score anomaly detection per parameter
    try:
        param_stats = df.groupby("Parameter")["Value"].agg(["mean", "std"]).rename(columns={"mean": "m", "std": "s"})
        df = df.join(param_stats, on="Parameter")
        df["z"] = (df["Value"] - df["m"]) / (df["s"].replace(0, np.nan))
        anomalies = df[(df["z"].abs() >= 2) & df["z"].notna()]
        if not anomalies.empty:
            top = anomalies.sort_values("z", key=lambda s: s.abs(), ascending=False).head(3)
            for _, r in top.iterrows():
                out["alerts"].append(
                    f"Anomaly: {r['Parameter']} in {r['City']} at {r['Value']} {r['Unit']} (z={r['z']:.1f})."
                )
    except Exception:
        pass

    # Deltas vs previous snapshot (city averages)
    if plan.get("compute_deltas") and st.session_state.prev_snapshot is not None and not df.empty:
        prev = st.session_state.prev_snapshot.copy()
        prev["Value"] = pd.to_numeric(prev["Value"], errors="coerce")
        prev = prev.dropna(subset=["Value"])
        prev_avg = prev.groupby("City")["Value"].mean()
        cur_avg = df.groupby("City")["Value"].mean()
        deltas = (cur_avg - prev_avg).dropna().sort_values(ascending=False)
        if not deltas.empty:
            out["deltas"] = deltas

    out["df"] = df
    return out

# -----------------------------
# LOADER (renders UI choices)
# -----------------------------
def loader_show(output: dict):
    df = output["df"]
    if df.empty:
        st.warning("No data available. Try another country code.")
        return

    # Live snapshot
    st.markdown("### 📊 Live Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Measurements", f"{len(df):,}")
    c2.metric("Cities", f"{df['City'].nunique():,}")
    try:
        last_ts = pd.to_datetime(df["Last Updated"]).max()
        c3.metric("Last Updated", last_ts.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        c3.metric("Last Updated", "—")

    # Table
    st.markdown("### 🗂️ Latest Measurements")
    st.dataframe(df[["City", "Parameter", "Value", "Unit", "Last Updated"]], use_container_width=True, height=360)

    # Charts the planner suggested
    if output["avg_by_city"] is not None and not output["avg_by_city"].empty:
        st.markdown("### 📈 Average Value by City")
        st.bar_chart(output["avg_by_city"])

    if output["avg_by_param"] is not None and not output["avg_by_param"].empty:
        st.markdown("### 🧪 Average Value by Parameter")
        st.bar_chart(output["avg_by_param"])

    if output["deltas"] is not None and not output["deltas"].empty:
        st.markdown("### 🔺 Change vs Previous Fetch (City Avg)")
        st.bar_chart(output["deltas"])

# -----------------------------
# CRITIC (checks coherence)
# -----------------------------
def critic(plan: dict, output: dict) -> list:
    notes = []
    if output["df"].empty:
        notes.append("Critic: Pipeline executed but returned no rows. Check country code or API availability.")
    if plan.get("aggregate_city") and (output["avg_by_city"] is None or output["avg_by_city"].empty):
        notes.append("Critic: Planner chose city aggregation but it produced no values.")
    if plan.get("aggregate_param") and (output["avg_by_param"] is None or output["avg_by_param"].empty):
        notes.append("Critic: Planner chose parameter aggregation but it produced no values.")
    if output["alerts"]:
        notes.append("Critic: Anomalies detected; verify sensor reliability or local events.")
    return notes

# -----------------------------
# ANIMATED ETL BANNER
# -----------------------------
st.markdown("### 🔄 Agentic ELT Pipeline")
st.markdown(
    f"""
    <div style="font-size:1.1rem;padding:.6rem 1rem;border-radius:1rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;">
        <b>{animated_etl()}</b><br/>
        <span style="font-size:.9rem;opacity:.85;">API ➜ Clean & Aggregate ➜ Visualize & Persist (short-term memory)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# RUN PIPELINE
# -----------------------------
# Extract
df_raw = fetch_openaq(country, limit)

# Plan
plan = planner(df_raw)

# Transform
out = transformer(df_raw, plan)

# Load (render)
loader_show(out)

# Agents speak (human-like, data-driven)
st.markdown("### 🤖 Agent Commentary")
agent_msgs = []

if out["df"].empty:
    agent_msgs.append("🧑‍💻 Data Engineer: No payload received. Will retry; consider changing country code.")
    agent_msgs.append("📊 Data Scientist: Without observations, trend checks are paused.")
    agent_msgs.append("🏛️ Policy Advisor: No health/context guidance until sensors report.")
else:
    cities = out["df"]["City"].nunique()
    params = out["df"]["Parameter"].nunique()
    agent_msgs.append(f"🧑‍💻 Data Engineer: Pulled **{len(out['df'])}** rows from OpenAQ across **{cities}** cities and **{params}** parameters.")
    if out["avg_by_city"] is not None and not out["avg_by_city"].empty:
        top_city = out["avg_by_city"].index[0]
        top_val = out["avg_by_city"].iloc[0]
        agent_msgs.append(f"📊 Data Scientist: Highest average reading right now is **{top_val:.2f}** in **{top_city}**.")
    else:
        agent_msgs.append("📊 Data Scientist: City averages not decisive; data too sparse or single-city sample.")

    if out["alerts"]:
        agent_msgs.append("🏥 Health Analyst: " + " ".join(out["alerts"][:2]))
    else:
        agent_msgs.append("🏥 Health Analyst: No strong anomalies detected; continue routine monitoring.")

for m in agent_msgs:
    st.info(m)
keep_memory(agent_msgs[-1] if agent_msgs else "No insight.")

# Critic notes
critic_notes = critic(plan, out)
if critic_notes:
    with st.expander("🧪 Critic QA Notes"):
        for n in critic_notes:
            st.warning(n)

# Memory panel (short-term)
with st.expander("🧠 Short-Term Memory (recent insights)"):
    if st.session_state.mem_insights:
        for i, m in enumerate(reversed(st.session_state.mem_insights), 1):
            st.write(f"{i}. {m}")
    else:
        st.write("No memories yet.")

# Save snapshot for deltas next run
st.session_state.prev_snapshot = out["df"][["City", "Parameter", "Value", "Unit", "Last Updated"]].copy() if not out["df"].empty else None

# -----------------------------
# HOW TO USE – 15 LINES
# -----------------------------
with st.expander("ℹ️ How to Use This App", expanded=True):
    st.markdown("""
    1. Open the app in your browser (Streamlit Cloud).
    2. This is a real-time **Agentic ELT** on open data (OpenAQ).
    3. Use the sidebar to enter a country code (e.g., `IN`, `US`, `GB`).
    4. Choose how many cities to fetch (5–50).
    5. Set cache TTL (seconds) to balance freshness vs. API calls.
    6. The **ELT banner** shows animated flow across Extract → Transform → Load.
    7. **Planner** decides aggregations and whether to compute deltas.
    8. **Transformer** cleans data, aggregates, and flags anomalies (z-scores).
    9. **Loader** renders tables and charts the planner requested.
    10. **Critic** checks coherence and flags issues.
    11. **Agents** (Engineer/Scientist/Health) narrate insights based on live data.
    12. Memory stores recent insights (short-term), updated each run.
    13. Use different country codes to compare regions.
    14. Increase TTL to reduce calls; decrease for fresher data.
    15. 100% free: no keys, no DB—ideal for demos & portfolios.
    """)

# Footer and manual refresh
st.markdown("---")
c1, c2 = st.columns([1, 6])
with c1:
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.experimental_rerun()
with c2:
    st.caption("Source: OpenAQ • This is a lightweight, rule-based agentic pipeline designed for Streamlit free tier.")
