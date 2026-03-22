import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Loading momentum data")
df = pl.read_parquet("model_ready_momentum.parquet")

# 1. Labeling
win_logic = (
    df.group_by("match_id")
    .agg([pl.col("runs_to_win").min().alias("final_deficit")])
    .with_columns(pl.when(pl.col("final_deficit") <= 0).then(1).otherwise(0).alias("result"))
)
df = df.join(win_logic.select(["match_id", "result"]), on="match_id", how="inner")

# 2. Features (Included Momentum)
features = ["ball", "current_score", "wickets_left", "crr", "rrr", 
            "runs_to_win", "balls_left", "last_18_runs", "last_18_wickets"]

X = df.select(features).to_pandas()
y = df.select("result").to_pandas()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost ")
model = xgb.XGBClassifier(
    n_estimators=300,        # More trees
    max_depth=8,             # Deeper trees
    learning_rate=0.05,      # Careful learning
    tree_method='hist',
    random_state=42
)

model.fit(X_train, y_train)

# 3. Save
joblib.dump(model, "cricket_win_model_v2.pkl")
joblib.dump(features, "model_features_v2.pkl")

print(f"NEW ACCURACY: {accuracy_score(y_test, model.predict(X_test)):.2%}")