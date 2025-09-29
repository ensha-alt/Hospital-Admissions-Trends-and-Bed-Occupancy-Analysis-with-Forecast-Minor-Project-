🏥 Hospital Admissions Trends & Bed Occupancy Analysis with Forecast Dashboard

📌 Project Overview: 
This project is my Minor Project (BCA), built using a synthetic hospital dataset. The goal was to analyze hospital admissions, bed occupancy, stay durations, and patient demographics, and to build predictive models for future admissions and occupancy.

The project demonstrates the complete analytics cycle:
📊 Data Cleaning & Feature Engineering
📈 Exploratory Data Analysis (EDA)
🔍 Statistical & Predictive Modeling (Prophet, ARIMA)
📉 Interactive Dashboards with Power BI
🎯 Data Storytelling & Recommendations

⚙️ Tech Stack: 
Python: Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Prophet
Visualization: Power BI (interactive dashboards)
Environment: Jupyter Notebook / VS Code
Dataset: Synthetic dataset (hospital_processed_with_forecasts.csv)

📊 Dashboard Insights: 
The Power BI dashboard consists of 4 sheets:
Executive Summary → KPIs (Total Admissions: 1,787 | Avg Bed Occupancy: 60.47% | Avg Stay Duration: 7.86 days)
Department Analysis → Bed occupancy & stay duration by department
Patient Demographics → Age group & gender distribution, stay duration trends
Forecasting → Actual vs Forecasted admissions & bed occupancy (Prophet vs ARIMA)

📌 Prophet model outperformed ARIMA (RMSE: 1.7 vs 10.4).

🔮 Key Results: 
Avg Bed Occupancy: 60.47%
Avg Stay Duration: 7.86 days
Forecasted Admissions: ~5.6K patients (next period)
High-load departments: Cardiology, Orthopedics, Neurology
Longer stays observed in elderly patients (60+)

🎯 Recommendations: 
Allocate more resources to Emergency & Surgery (high occupancy + long stays).
Plan ahead for forecasted admissions surge (~5.6K).
Develop elderly care capacity due to longer stays in 60+ patients.
Use predictive insights for staff scheduling & capacity planning.

🚀 Future Scope (Major Project Extension): 
Replace synthetic data with real hospital dataset.
Deploy dashboard on Power BI Service / Cloud for real-time updates.
Add SQL database backend for live integration.
Enhance security with role-based access.
