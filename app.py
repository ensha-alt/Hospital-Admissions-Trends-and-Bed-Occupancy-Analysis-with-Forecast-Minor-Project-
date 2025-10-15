# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from prophet import Prophet

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Hospital Dashboard", layout="wide", page_icon="🏥")

# ---------- SESSION STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------- LOGIN PAGE ----------
def login_page():
    st.markdown("""
    <h2 style='text-align:center; color:#0C2340;'>🏥 Hospital Management Dashboard</h2>
    <h4 style='text-align:center; color:gray;'>Please login to continue</h4>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter hospital admin name")
        password = st.text_input("Password", placeholder="Enter password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            # Demo credentials - you can replace with your own list/dictionary
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.success("Login successful! Redirecting to dashboard...")
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.markdown("""
    <div style='text-align:center; margin-top:30px;'>
        <small>Demo credentials → <b>Username:</b> admin | <b>Password:</b> 1234</small>
    </div>
    """, unsafe_allow_html=True)

# ---------- DASHBOARD PAGE ----------
def dashboard_page():
    st.title("🏥 Hospital Admissions & Bed Occupancy Dashboard")

    # --- Load Data ---
    @st.cache_data
    def load_data(path="hospital_processed_with_forecasts.csv"):
        df = pd.read_csv(path, parse_dates=["Admission_Date","Discharge_Date","Date"], dayfirst=False)
        numeric_cols = ["Bed_Occupancy_Rate","Admissions_Forecast","Admissions_Forecast_Lower","Admissions_Forecast_Upper",
                        "Occupancy_Forecast","Occupancy_Forecast_Lower","Occupancy_Forecast_Upper"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Dataset missing! Please place hospital_processed_with_forecasts.csv in the folder.")
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    dept_list = np.sort(df["Department"].dropna().unique()) if "Department" in df.columns else []
    selected_dept = st.sidebar.selectbox("Select Department", np.concatenate((["All"], dept_list)))
    date_min, date_max = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input("Date Range", (date_min, date_max))
    show_forecast_conf = st.sidebar.checkbox("Show Confidence Bands", True)

    # --- Apply filters ---
    filtered = df.copy()
    if selected_dept != "All":
        filtered = filtered[filtered["Department"] == selected_dept]
    filtered = filtered[(filtered["Date"] >= pd.to_datetime(date_range[0])) & (filtered["Date"] <= pd.to_datetime(date_range[1]))]

    # --- KPI Function ---
    def kpi_data(df):
        latest_date = df["Date"].max()
        latest = df[df["Date"] == latest_date]
        return {
            "latest_date": latest_date,
            "total_patients": df["Patient_ID"].nunique(),
            "beds_occupied": latest["Beds_Occupied"].sum(),
            "avg_occupancy": df["Bed_Occupancy_Rate"].mean(),
            "avg_stay": df["Length_of_Stay_Days"].mean()
        }

    kpis = kpi_data(filtered)

    # --- Tabs (Sheets) ---
    tabs = st.tabs(["Executive Summary", "Department Analysis", "Patient Demographics", "Forecasting", "Key Insights"])

    # ----------- EXECUTIVE SUMMARY -----------
    with tabs[0]:
        st.subheader("📈 Executive Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Date", str(kpis["latest_date"].date()))
        c2.metric("Total Patients", kpis["total_patients"])
        c3.metric("Beds Occupied", int(kpis["beds_occupied"]))
        c4.metric("Avg Occupancy (%)", f"{kpis['avg_occupancy']:.1f}")

        # Line charts
        adm = filtered.groupby("Date")["Patient_ID"].nunique().reset_index()
        occ = filtered.groupby("Date")["Bed_Occupancy_Rate"].mean().reset_index()
        fig1 = px.line(adm, x="Date", y="Patient_ID", title="Admissions Over Time")
        fig2 = px.line(occ, x="Date", y="Bed_Occupancy_Rate", title="Bed Occupancy Over Time")
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

    # ----------- DEPARTMENT ANALYSIS -----------
    with tabs[1]:
        st.subheader("🏨 Department Analysis")
        dept_grp = filtered.groupby("Department").agg(
            Admissions=("Patient_ID","nunique"),
            Avg_Occupancy=("Bed_Occupancy_Rate","mean"),
            Avg_LOS=("Length_of_Stay_Days","mean")
        ).reset_index()
        fig = px.bar(dept_grp, x="Admissions", y="Department", orientation="h", title="Admissions by Department")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dept_grp.round(2))

    # ----------- PATIENT DEMOGRAPHICS -----------
    with tabs[2]:
        st.subheader("👩‍⚕️ Patient Demographics")
        if "Gender" in filtered.columns:
            figg = px.pie(filtered, names="Gender", title="Gender Distribution")
            st.plotly_chart(figg, use_container_width=True)
        if "Age_Group" in filtered.columns:
            fig_age = px.bar(filtered, x="Age_Group", title="Age Group Distribution")
            st.plotly_chart(fig_age, use_container_width=True)

    # ----------- FORECASTING (with Prophet) -----------
    with tabs[3]:
        st.subheader("📊 Forecasting (Prophet-based)")
        st.caption("Regenerate forecast for admissions and occupancy dynamically.")

        if st.button("🔄 Regenerate Forecasts"):
            st.info("Training Prophet models...")
            try:
                # Admissions Forecast
                adm_ts = filtered.groupby("Date")["Patient_ID"].nunique().reset_index(name="y")
                adm_ts.rename(columns={"Date":"ds"}, inplace=True)
                model = Prophet()
                model.fit(adm_ts)
                future = model.make_future_dataframe(periods=30)
                fc = model.predict(future)

                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(x=adm_ts["ds"], y=adm_ts["y"], mode="lines+markers", name="Actual"))
                fig_a.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Forecast"))
                if show_forecast_conf:
                    fig_a.add_trace(go.Scatter(
                        x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                        y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                        fill="toself", fillcolor="rgba(0,176,246,0.2)",
                        line=dict(color="rgba(255,255,255,0)"), showlegend=True, name="Confidence"
                    ))
                fig_a.update_layout(title="Prophet Forecast — Admissions")
                st.plotly_chart(fig_a, use_container_width=True)
            except Exception as e:
                st.error(f"Forecast error: {e}")

    # ----------- KEY INSIGHTS -----------
    with tabs[4]:
        st.subheader("💡 Key Insights")
        st.write(f"- **Top Department:** {filtered['Department'].mode()[0] if 'Department' in filtered.columns else 'N/A'}")
        st.write(f"- **Average Stay Duration:** {kpis['avg_stay']:.1f} days")
        st.write(f"- **Average Occupancy Rate:** {kpis['avg_occupancy']:.1f}%")
        st.write(f"- **Total Patients:** {kpis['total_patients']}")

    st.markdown("---")
    st.caption("Developed by Insha Farhan | Hospital Admissions & Bed Occupancy Analysis")

# ---------- MAIN APP FLOW ----------
if not st.session_state.logged_in:
    login_page()
else:
    dashboard_page()
