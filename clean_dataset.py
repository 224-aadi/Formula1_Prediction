import pandas as pd

p = 'data/processed/dataset_seasons_2024_fastf1.parquet'
df = pd.read_parquet(p)
for c in ['q_s1_gap', 'q_s2_gap', 'q_s3_gap', 'fp_longrun_gap']:
    df[c] = df.get(c + '_x', df.get(c))
    df.drop(columns=[x for x in [c + '_x', c + '_y'] if x in df.columns], inplace=True, errors='ignore')
df.to_parquet(p, index=False)
print('cleaned', p)
print([c for c in df.columns if 'q_s' in c or 'fp' in c])