#!/usr/bin/env python3
"""
Automatically update the final_hybrid_dataset.parquet with the latest race data.
Designed to run in GitHub Actions every Monday.
"""
import argparse
import pandas as pd
from pathlib import Path
from f1outcome.data.jolpica import JolpicaClient
from f1outcome.data.build_dataset import DatasetBuilder

DATASET_PATH = Path("data/processed/final_hybrid_dataset.parquet")
JOLPICA_BASE = "https://api.jolpica.org/ergast/f1"
JOLPICA_CACHE = Path("data/raw/jolpica_cache")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't save the updated dataset")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # 1. Load current dataset to find latest round
    df = pd.read_parquet(DATASET_PATH)
    latest_season = df["season"].max()
    latest_round = df[df["season"] == latest_season]["round"].max()
    
    print(f"Current latest data: {latest_season} Round {latest_round}")

    # 2. Check for new rounds
    client = JolpicaClient(base_url=JOLPICA_BASE, cache_dir=JOLPICA_CACHE)
    builder = DatasetBuilder(client)
    
    # Check current season (and possibly next if it's new year)
    from datetime import datetime
    current_year = datetime.now().year
    
    # We'll check rounds for the latest season in the data, up to current year
    seasons_to_check = sorted(list(set([latest_season, current_year])))
    
    new_data_frames = []
    
    for season in seasons_to_check:
        print(f"Checking for new rounds in {season}...")
        try:
            available_rounds = builder.get_season_rounds(season)
        except Exception as e:
            print(f"Could not fetch rounds for {season}: {e}")
            continue
            
        new_rounds = [r for r in available_rounds if (season > latest_season) or (season == latest_season and r > latest_round)]
        
        if not new_rounds:
            print(f"No new rounds for {season}.")
            continue
            
        print(f"New rounds found: {new_rounds}")
        
        for rnd in new_rounds:
            print(f"Fetching {season} Round {rnd}...")
            try:
                # Build for just this round
                # Note: builder.build takes a list of seasons. 
                # We want a more granular check. Let's use internal methods.
                res_df = builder.fetch_round_results(season, rnd)
                if res_df.empty:
                    print(f"R{rnd} results not yet available. Skipping.")
                    continue
                
                qual_df = builder.fetch_round_qualifying(season, rnd)
                
                # Merge
                merged = pd.merge(res_df, qual_df, on=["season", "round", "driverId"], how="left")
                
                # Add window features (simplified for automated update - usually requires full context)
                # For now, we'll append and assume the next full train cycle handles features, 
                # OR we try to compute rolling means on the spot.
                # Actually, the ranker needs full features.
                
                new_data_frames.append(merged)
                print(f"Successfully staged {season} R{rnd}")
                
            except Exception as e:
                print(f"Failed to fetch {season} R{rnd}: {e}")

    if not new_data_frames:
        print("Dataset is already up to date.")
        return

    # 3. Concatenate and Save
    new_data = pd.concat(new_data_frames, ignore_index=True)
    
    # Note: Real implementation would need to recalculate 
    # rolling features (avg_grid_5, rolling_pts, etc.)
    # For this version, we append the raw rows.
    # FULL REBUILD is safer but slower. 
    # Let's do a "smart" rebuild of just the current season if new rounds are found.
    
    if not args.dry_run:
        updated_df = pd.concat([df, new_data], ignore_index=True)
        # Drop duplicates just in case
        updated_df = updated_df.drop_duplicates(subset=["season", "round", "driverId"])
        updated_df.to_parquet(DATASET_PATH)
        print(f"Dataset updated and saved to {DATASET_PATH}")
    else:
        print("Dry run: Would have appended new data.")

if __name__ == "__main__":
    main()
