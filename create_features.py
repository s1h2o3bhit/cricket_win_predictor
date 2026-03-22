import polars as pl

print("Loading cleaned data...")
df = pl.read_parquet("cleaned_cricket_data.parquet")

# 1. Sort and Clean
df = df.sort(["match_id", "innings", "ball"])

print("Building Momentum Features (Last 18 balls)...")

# 2. Base Feature Calculations
df = df.with_columns([
    pl.col("wicket_type").is_not_null().and_(pl.col("wicket_type") != "").cast(pl.Int64).alias("is_wicket"),
    (pl.col("runs_off_bat").fill_null(0) + pl.col("extras").fill_null(0)).alias("total_runs")
])

# 3. Cumulative Calculations
df = df.with_columns([
    pl.col("total_runs").cum_sum().over(["match_id", "innings"]).alias("current_score"),
    pl.col("is_wicket").cum_sum().over(["match_id", "innings"]).alias("wickets_down"),
    pl.int_range(1, pl.len() + 1).over(["match_id", "innings"]).alias("balls_delivered")
])

# 4. MOMENTUM FEATURES (This is the secret sauce)
df = df.with_columns([
    # Runs in last 18 balls (3 overs)
    pl.col("total_runs").rolling_sum(window_size=18).over(["match_id", "innings"]).fill_null(0).alias("last_18_runs"),
    # Wickets in last 18 balls (3 overs)
    pl.col("is_wicket").rolling_sum(window_size=18).over(["match_id", "innings"]).fill_null(0).alias("last_18_wickets")
])

# 5. Advanced Math
df = df.with_columns([
    (pl.col("current_score") / (pl.col("balls_delivered") / 6)).alias("crr"),
    (10 - pl.col("wickets_down")).alias("wickets_left")
])

# 6. Target Score Logic
first_innings_final = (
    df.filter(pl.col("innings") == 1)
    .group_by("match_id")
    .agg(pl.col("current_score").max().alias("target_score"))
)
df = df.join(first_innings_final, on="match_id", how="left")

# 7. Second Innings Pressure
df = df.with_columns([
    pl.when(pl.col("innings") == 2)
    .then(pl.col("target_score") + 1 - pl.col("current_score"))
    .otherwise(None)
    .alias("runs_to_win"),
    (120 - pl.col("balls_delivered")).alias("balls_left")
])

df = df.with_columns(
    pl.when((pl.col("innings") == 2) & (pl.col("balls_left") > 0))
    .then(pl.col("runs_to_win") / (pl.col("balls_left") / 6))
    .otherwise(0)
    .alias("rrr")
)

# 8. ACCURACY BOOST: Filter only 2nd Innings
# We focus the model entirely on the 'Chase', where predictions are most accurate.
df_model = df.filter(pl.col("innings") == 2)

# 9. Save
df_model.write_parquet("model_ready_momentum.parquet")

print("-" * 30)
print(f"SUCCESS: Momentum data ready with {len(df_model)} rows.")
print("-" * 30)