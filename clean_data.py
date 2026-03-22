import polars as pl

# 1. Load the master parquet file
df = pl.read_parquet("final_cricket_data.parquet")

print("Original Schema:", df.schema)

# 2. Cleaning & Type Conversion
# We need to identify which columns represent runs, wickets, and dates.
# Note: Column names depend on your specific dataset (e.g., Cricsheet vs Kaggle)
# Assuming standard ball-by-ball column names:
df_cleaned = df.with_columns([
    # Convert numeric columns (using 'strict=False' to turn non-numbers into Null)
    pl.col("ball").cast(pl.Float64, strict=False),
    pl.col("runs_off_bat").cast(pl.Int64, strict=False),
    pl.col("extras").cast(pl.Int64, strict=False),
    
    # Convert dates
    pl.col("start_date").str.to_date(strict=False)
])

# 3. Handle Missing Values
# Drop rows where essential data (like runs or ball number) is missing
df_cleaned = df_cleaned.drop_nulls(subset=["ball", "runs_off_bat"])

# 4. Feature Engineering: Create a 'Total Runs' column per ball
df_cleaned = df_cleaned.with_columns(
    (pl.col("runs_off_bat") + pl.col("extras")).alias("total_runs_delivery")
)

# 5. Save the CLEANED version
df_cleaned.write_parquet("cleaned_cricket_data.parquet")

print(f"Cleaned data saved! New Row Count: {len(df_cleaned)}")
print("New Schema:", df_cleaned.schema)