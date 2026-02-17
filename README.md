# COMP3610 Assignment 1

## Overview
This project builds an end-to-end data pipeline and interactive dashboard using the **NYC TLC Yellow Taxi Trip Records (January 2024)** dataset.

### What this project does
1. **Programmatically downloads** the required TLC datasets into `data/raw/`
2. **Validates** schema and datatypes (required columns + datetime checks)
3. **Cleans** invalid / null trips and records removal counts
4. **Feature engineers** exactly 4 derived columns:
   - `trip_duration_minutes`
   - `trip_speed_mph`
   - `pickup_hour`
   - `pickup_day_of_week`
5. Loads the cleaned dataset into **DuckDB** and runs **exactly 5 SQL analysis queries**
6. Builds a **Streamlit dashboard** with **exactly 5 interactive visualizations** (Plotly), driven by filters

---

## Required Data Sources
This assignment uses exactly the following files (downloaded via Python in the notebook):

- Yellow Taxi Trip Data (January 2024):  
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet

- Taxi Zone Lookup Table:  
  https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

---

## Repository Structure
```text
.
├── assignment1.ipynb        # Jupyter notebook (Parts 1, 2, 3 prototypes)
├── app.py                   # Streamlit dashboard main page
├── pages/
│   ├── 1_Overview.py        # Data overview, stats, column info, quality checks
│   ├── 2_Visualizations.py  # All 5 interactive Plotly charts + filters
│   └── 3_Upload_Data.py     # Upload your own CSV and build charts
├── data/
│   ├── raw/                 (downloaded files — not committed)
│   └── processed/           (cleaned parquet — not committed)
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Guide
### 1) Create and activate a virtual environment

**Windows (Powershell)**
```bash 
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
<u>N.B.</u> pip-26.0.1 is the pip version used & kernal Python 3.12.7 used
```bash
pip install -r requirements.txt
```

### 3) Run the pipeline
Open and run assignment1.ipynb 
This will generate:
 - data/raw/ (downloaded datasets)
 - data/processed/yellow_2024_01_clean.parquet (cleaned output)

 ### 4) Run the Streamlit dashboard
 ```bash
 streamlit run app.py
 ```

 ### 5) Deactivate the virtual environment (when finished)

 ** Windows/macOS/Linux
 ```bash 
 deactivate
 ```

## License
This project is licensed under Samuel Soman, student at the University of the West Indies.

