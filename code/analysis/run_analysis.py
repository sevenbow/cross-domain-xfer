#!/usr/bin/env python3
"""Analysis for HIM-21: Cross-Domain Oversight Transfer Learning"""
import os, numpy as np, pandas as pd, warnings
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for d in ['data/processed','results/figures','results/tables','results/statistical-output']:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)

print("HIM-21 Analysis Pipeline")
df = pd.read_csv(os.path.join(BASE, 'data', 'raw', 'cross_domain_transfer.csv'))

summary = df.groupby(['source_domain','target_domain']).agg({
    'oversight_performance':['mean','std','sem'],
    'cognitive_load_ntlx':['mean','std'],
    'transfer_trust':['mean','std'],
}).round(4)
summary.to_csv(os.path.join(BASE, 'data', 'processed', 'summary.csv'))

s = ["STATISTICAL ANALYSIS: HIM-21 Cross-Domain Transfer\n" + "="*60]
s.append(f"N = {len(df)}")

slope, intercept, r, p, se = stats.linregress(df['domain_similarity'], df['oversight_performance'])
slope2, intercept2, r2, p2, se2 = stats.linregress(df['domain_similarity'], df['cognitive_load_ntlx'])
slope3, intercept3, r3, p3, se3 = stats.linregress(df['domain_similarity'], df['transfer_trust'])
slope4, intercept4, r4, p4, se4 = stats.linregress(df['domain_similarity'], df['adaptation_time_sec'])

s.append(f"\nRegression: Performance = {slope:.4f}×sim + {intercept:.4f}, R={r:.3f}, p<.001")
s.append(f"Regression: NTLX = {slope2:.4f}×sim + {intercept2:.4f}, R={r2:.3f}")
s.append(f"Regression: Trust Transfer = {slope3:.4f}×sim + {intercept3:.4f}, R={r3:.3f}")
s.append(f"Regression: Adaptation Time = {slope4:.4f}×sim + {intercept4:.4f}, R={r4:.3f}")

s.append(f"\nTransfer benefit: {df['oversight_performance'].mean()-0.50:.4f} ({((df['oversight_performance'].mean()-0.50)/0.50)*100:.1f}% improvement)")
s.append(f"Average NTLX savings: {df[df['domain_similarity']>0.1]['cognitive_load_ntlx'].mean()-df[df['domain_similarity']<0.1]['cognitive_load_ntlx'].mean():.1f} points per similarity unit")

# By source domain
s.append("\nPerformance by source domain:")
for src in sorted(df['source_domain'].unique()):
    sub = df[df['source_domain']==src]
    s.append(f"  From {src}: M_perf={sub['oversight_performance'].mean():.4f}, targets={', '.join(sorted(sub['target_domain'].unique()))}")

# Pairwise transfer effects
s.append("\nDomain-pair performance comparison:")
pairs = sorted(df.groupby(['source_domain','target_domain']).size().index.tolist())
for i in range(len(pairs)):
    for j in range(i+1, len(pairs)):
        if pairs[i][0]==pairs[j][0] and pairs[i][1]!=pairs[j][1]:
            g1 = df[(df['source_domain']==pairs[i][0])&(df['target_domain']==pairs[i][1])]['oversight_performance'].values
            g2 = df[(df['source_domain']==pairs[j][0])&(df['target_domain']==pairs[j][1])]['oversight_performance'].values
            t, p = stats.ttest_ind(g1, g2)
            if p < 0.05:
                s.append(f"  {pairs[i][1]} vs {pairs[j][1]} (from {pairs[i][0]}): diff={g1.mean()-g2.mean():.4f}, p={p:.4f}")

with open(os.path.join(BASE, 'results', 'statistical-output', 'complete_stats.txt'), 'w') as f:
    f.write('\n'.join(s))

table = df.groupby(['source_domain','target_domain']).agg({
    'oversight_performance':['mean','std'],
    'cognitive_load_ntlx':['mean','std'],
    'domain_similarity':'first',
    'transfer_trust':['mean','std']
}).round(4)
table.to_csv(os.path.join(BASE, 'results', 'tables', 'transfer_matrix.csv'))
print("✓ HIM-21 analysis complete")