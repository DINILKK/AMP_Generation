import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset5_TransImbAMP.csv')  # change filename

strain_cols = ['Acinetobacter baumannii ATCC 19606',
               'Escherichia coli ATCC 25922',
               'Klebsiella pneumoniae ATCC 700603',
               'Pseudomonas aeruginosa ATCC 27853',
               'Staphylococcus aureus ATCC 29213',
               'Staphylococcus aureus ATCC 33591',
               'Staphylococcus aureus ATCC43300',
               'Staphylococcus aureus USA300']

df['is_AMP'] = df[strain_cols].max(axis=1)
seqs = df[df['is_AMP'] == 1][['SEQUENCE']]

# Filter sequences longer than 126 amino acids
before = len(seqs)
seqs = seqs[seqs['SEQUENCE'].str.len() <= 126]
after = len(seqs)
print(f"Removed {before - after} sequences longer than 126 AA")
print(f"Remaining: {after}")

train, val = train_test_split(seqs, test_size=0.2, random_state=42)
train.to_csv('./data/LatentDiffusion_Train', index=False)
val.to_csv('./data/LatentDiffusion_Val', index=False)
print(f"Train: {len(train)}, Val: {len(val)}")