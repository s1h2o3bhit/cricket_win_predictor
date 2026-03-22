import polars as pl
import os
import glob
import csv

# 1. Configuration
input_folder = r"D:\cricket analysis project" 
output_file = "cleaned_cricket_data.parquet"

files = glob.glob(os.path.join(input_folder, "*.csv"))
print(f"Found {len(files)} files. Starting extraction...")

all_ball_data = []

# 2. Extraction Loop
for i, f_path in enumerate(files):
    try:
        match_id = os.path.basename(f_path).replace('.csv', '')
        
        with open(f_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                # Check if row starts with 'ball'
                if row and row[0].strip() == 'ball':
                    # Ensure the row has enough columns for wicket data (at least 16)
                    if len(row) >= 16:
                        all_ball_data.append([
                            match_id,
                            row[1],   # innings
                            row[2],   # ball_no
                            row[3],   # batting_team
                            row[4],   # batter
                            row[6],   # bowler
                            row[7],   # runs_off_bat
                            row[8],   # extras
                            row[14],  # wicket_type (NEW)
                            row[15]   # player_out (NEW)
                        ])
                    # If row is shorter but still a 'ball' row, pad with empty strings
                    elif len(row) >= 9:
                        all_ball_data.append([
                            match_id, row[1], row[2], row[3], row[4], 
                            row[6], row[7], row[8], "", ""
                        ])
        
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1}/{len(files)} files...")
            
    except Exception as e:
        continue

# 3. Final Processing
if not all_ball_data:
    print("Error: No data extracted. Check your file path or CSV format.")
else:
    print(f"Extracted {len(all_ball_data)} rows. Converting to optimized format...")
    
    # Create the Polars DataFrame with the updated schema
    schema = [
        "match_id", "innings", "ball", "batting_team", 
        "batter", "bowler", "runs_off_bat", "extras", 
        "wicket_type", "player_out"
    ]
    
    df = pl.DataFrame(all_ball_data, schema=schema, orient="row")
    
    # 4. Data Cleaning & Type Casting
    df = df.with_columns([
        pl.col("innings").cast(pl.Int64, strict=False),
        pl.col("ball").cast(pl.Float64, strict=False),
        pl.col("runs_off_bat").cast(pl.Int64, strict=False).fill_null(0),
        pl.col("extras").cast(pl.Int64, strict=False).fill_null(0),
    ])
    
    # Calculate Total Runs per ball
    df = df.with_columns(
        (pl.col("runs_off_bat") + pl.col("extras")).alias("total_runs")
    )
    
    # Save as Parquet
    df.write_parquet(output_file)
    print("-" * 30)
    print(f"SUCCESS! {output_file} is ready with WICKET data.")
    print(f"Total Rows: {len(df)}")
    print("-" * 30)