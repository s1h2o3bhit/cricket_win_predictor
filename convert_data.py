import polars as pl
import os
import time
import glob

input_folder = r"D:\cricket analysis project" 
output_file = "final_cricket_data.parquet"

def combine_csvs():
    start_time = time.time()
    search_pattern = os.path.join(input_folder, "*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        print("No CSV files found! Check your folder path.")
        return

    print(f"Found {len(files)} files. Starting robust merge...")
    
    df_list = []
    for i, f in enumerate(files):
        try:
            # We use truncate_ragged_lines to handle the metadata rows
            # We set infer_schema_length=0 to read everything as text to avoid crashes
            temp_df = pl.read_csv(
                f, 
                ignore_errors=True, 
                truncate_ragged_lines=True,
                infer_schema_length=0 
            )
            
            # Add the filename so you know which match is which
            temp_df = temp_df.with_columns(pl.lit(os.path.basename(f)).alias("file_source"))
            
            df_list.append(temp_df)
            
            # Show progress every 1000 files
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(files)} files...")
                
        except Exception as e:
            continue # Skip files that are truly broken

    print("Aligning all files into one master table...")
    # 'diagonal' allows merging even if some files have different headers
    final_df = pl.concat(df_list, how="diagonal")
    
    print(f"Final Row Count: {len(final_df)}")
    
    # Save to Parquet
    final_df.write_parquet(output_file)
    
    duration = round(time.time() - start_time, 2)
    print(f"SUCCESS! Created {output_file} in {duration}s")

if __name__ == "__main__":
    combine_csvs()