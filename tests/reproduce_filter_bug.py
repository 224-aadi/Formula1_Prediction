import pandas as pd
import numpy as np

def reproduce():
    # Create a dataframe with mixed types for season and round
    data = {
        "season": [2019, "2019", 2020, "2020 "], # Mixed int and string, with whitespace
        "round": [1, "1", 2, "2"],
        "value": [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame types:")
    print(df.dtypes)
    print(df)

    # Attempt simple filter (simulate the bug)
    print("\n--- Attempting simple filter: season==2019 & round==1 ---")
    # This might fail or return partial results depending on exact pandas version/behavior, 
    # but with mixed types it's unreliable.
    try:
        filtered = df[(df["season"] == 2019) & (df["round"] == 1)]
        print(f"Filtered rows (simple): {len(filtered)}")
        print(filtered)
    except Exception as e:
        print(f"Filter failed: {e}")

    # Apply normalization fix
    print("\n--- Applying normalization fix ---")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    
    print("Normalized DataFrame types:")
    print(df.dtypes)
    print(df)

    # Attempt filter again
    print("\n--- Attempting filter after normalization ---")
    filtered_fixed = df[(df["season"] == 2019) & (df["round"] == 1)]
    print(f"Filtered rows (normalized): {len(filtered_fixed)}")
    print(filtered_fixed)

    assert len(filtered_fixed) == 2, "Normalization failed to capture all 2019 round 1 rows!"
    print("\nSUCCESS: Normalization fixed the filtering issue.")

if __name__ == "__main__":
    reproduce()
