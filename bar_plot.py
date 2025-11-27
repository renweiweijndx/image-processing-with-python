# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:54:37 2025

@author: renwe
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannotations.Annotator import Annotator

# ========= Load data =========
df = pd.read_excel("C:/Users/renwe/Desktop/MFI.xlsx")

# rename first column to Group
df = df.rename(columns={' ':'Group'})

# ========= Summary per group =========
summary = df.groupby('Group')['Mean'].agg(['mean','std','count'])
print(summary)

## bar plot in seaborn
plt.figure(figsize=(6,4))
ax = sns.barplot(data=df, x="Group", y="Mean", errorbar="sd", 
                 palette="colorblind",alpha=.6, capsize=0.2, errwidth=1.5)
plt.xticks(rotation=45, ha='right')

# ---- Add individual data points ----
sns.stripplot(x="Group", y="Mean", data=df,
              color='black', facecolors='none', edgecolor='black',
              linewidth=1.0, size=5, jitter=True, alpha=1.0, marker='o')
sns.despine(offset=0, trim=False)
plt.tight_layout()
plt.savefig("MFI.pdf", bbox_inches="tight")

## Oneway ANOVA with Tukey
model = ols('Mean ~ C(Group)', data=df).fit()
anova_table = anova_lm(model)
print("\n=== One-way ANOVA ===")
print(anova_table)

# ===== Tukey HSD Post-hoc =====
tukey = pairwise_tukeyhsd(endog=df['Mean'],
                          groups=df['Group'],
                          alpha=0.05)

print("\n=== Tukey HSD ===")
print(tukey.summary())

# 生成配对（Tukey 需要所有组间对比）
groups = df["Group"].unique()
pairs = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]

# 用 statannotations 标注 Tukey 显著性
#annotator = Annotator(ax, pairs, data=df, x="Group", y="Mean")
#annotator.configure(test=None, text_format="star", show_test_name=False)
#annotator.set_pvalues(tukey.pvalues)
#annotator.annotate()