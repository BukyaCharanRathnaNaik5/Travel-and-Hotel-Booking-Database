# Big Data Analytics on Travel and Hotel Booking Database

Apache Spark (PySpark) pipeline that cleans and analyses hotel booking records
(demand, ADR, revenue, channels, cancellations) and predicts booking cancellations.

## Structure
- `data/hotel_bookings.csv`  dataset (120,000 synthetic records, 17 columns, schema of the public Hotel Booking Demand dataset)
- `src/generate_data.py`  regenerates the synthetic dataset
- `src/hotel_analytics.py`  PySpark: load, clean, feature engineering, Spark SQL analysis, saves Parquet
- `src/cancellation_model.py`  PySpark MLlib: Logistic Regression and Random Forest
- `src/run_pandas_analysis.py`  pandas/scikit-learn version (no Spark); creates charts + `outputs/results.json`
- `outputs/`  charts and results from the pandas run
- `report/`  the project report (Word)

## Run (from this folder)
    pip install -r requirements.txt        # Spark also needs Java 11 or 17
    python src/hotel_analytics.py          # or: spark-submit src/hotel_analytics.py
    python src/cancellation_model.py
    python src/run_pandas_analysis.py      # quick run without Spark

## Using the real dataset
Download "Hotel booking demand" (Kaggle) and save it as `data/hotel_bookings.csv`.
The real file has more columns and a `reservation_status` column that leaks the label; the
scripts only use the 17 columns listed in the report. Rerun everything and update the report numbers.

## Note
The report figures come from `run_pandas_analysis.py` on the synthetic data. The PySpark scripts
were not executed in the environment where this project was prepared; test them on your machine.
