🏥 Hospital Admissions & Bed Occupancy Analytics
📌 Overview

Minor project using a synthetic hospital dataset to analyze admissions, bed occupancy, stay duration, and demographics. Built predictive models (Prophet, ARIMA) and designed an interactive Power BI dashboard with storytelling and recommendations.

⚙️ Tech Stack

Python: Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Prophet

Visualization: Power BI

Dataset: Synthetic (hospital_processed_with_forecasts.csv)

📊 Key Insights

Total Admissions: 1,787

Avg Bed Occupancy: 60.47%

Avg Stay Duration: 7.86 days

Forecasted Admissions: ~5.6K

High-load Depts: Cardiology, Orthopedics, Neurology

Prophet model outperformed ARIMA (RMSE 1.7 vs 10.4)

🎯 Recommendations

Allocate resources to Emergency & Surgery

Plan ahead for forecasted patient surge

Improve elderly care capacity (longer stays 60+)

🚀 Future Scope

Apply on real datasets

Deploy on Power BI Service / Cloud

Integrate with SQL for live data

📂 Repo Contents

Minorproject_Notebook.ipynb → Data prep, EDA, forecasting

hospital_processed_with_forecasts.csv → Clean dataset

Minor Project.pdf → Dashboard export
