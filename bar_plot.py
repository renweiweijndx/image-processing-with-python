# -*- coding: utf-8 -*-
"""
带误差棒的柱状图。
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


## Read files
df1 = pd.read_csv('C:/Users/renwe/Desktop/wt_mock.csv')
df2 = pd.read_csv('C:/Users/renwe/Desktop/wt_pit.csv')
df1 = df1.iloc[:, 1:]
df2 = df2.iloc[:, 1:]

## Select data from files
df1_sel = df1[['RawIntDen']]
df2_sel = df2[['RawIntDen']]
df1_sel.columns = ['WT_mock']
df2_sel.columns = ['WT_pit']

## Joint data into a new df
df = pd.concat([df1_sel, df2_sel], axis=1)

## Melt the data for seaborn
df_melted = df.melt(var_name='Condition', value_name='Value')

# Perform independent two-sample t-test between WT and KO
t_stat, p_value = ttest_ind(df['WT_mock'], df['WT_pit'], equal_var=False)  # Welch's t-test
t_stat, p_value

## Plot
plt.figure(figsize=(3, 4)) ## Adjust if needed
sns.barplot(data=df_melted, x='Condition', y='Value', ci='sd', capsize=0.1, palette='pastel', errwidth=1.5)
sns.stripplot(data=df_melted, x='Condition', y='Value', color='black', jitter=True, alpha=0.9, marker='o', size=6)

# Add significance bar
y_max = max(df.max()) * 1.05
plt.plot([0, 0, 1, 1], [y_max, y_max*1.02, y_max*1.02, y_max], color='black')
plt.text(0.5, y_max*1.03, '**', ha='center', va='bottom', fontsize=16)

plt.title('Bar Plot with Error Bars and Data Points')
plt.ylabel('MFI')
plt.xlabel('Condition')
plt.tight_layout()

# Save as PDF
annotated_output_path = 'D:/plot/barplot_with_significance.pdf'
plt.savefig(annotated_output_path, format='pdf')
plt.show()
