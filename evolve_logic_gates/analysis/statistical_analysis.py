import numpy as np 
import pandas as pd 
from scipy.stats import mannwhitneyu  # For the Wilcoxon rank-sum test


df = pd.read_csv("results.csv")
bonferroni_correction = 16
# within encodings
for gate in ["AND", "NAND", "OR", "XOR"]:
    for encoding in ['size', 'stiffness']:
        pair_1 = [True, True]
        if encoding == 'size': 
            pair_2 = [False, True]
        if encoding == "stiffness":
            pair_2 = [True, False]
        
        group1 = df[(df["gate"] == gate) & (df["encoding"] == encoding) & (df["stiffness"] == pair_2[0]) & (df["size"] == pair_2[1])] 
        group2 = df[(df["gate"] == gate) & (df["encoding"] == encoding) & (df["stiffness"] == True) & (df["size"] == True)]

        stat, p_value = mannwhitneyu(group1['fitness_gen_100'], group2['fitness_gen_100'])
        p_value*=bonferroni_correction
        print(f"gate {gate}, encoding {encoding}, p-value: {p_value}")

# across different encodings
for gate in ["AND", "NAND", "OR", "XOR"]:
    for conditions in [[True,True]]:
        group1 = df[(df["gate"] == gate) & (df["encoding"] == "size") & (df["stiffness"] == conditions[0]) & (df["size"] == conditions[1])] 
        group2 = df[(df["gate"] == gate) & (df["encoding"] == "stiffness") & (df["stiffness"] == conditions[0]) & (df["size"] == conditions[1])]

        stat, p_value = mannwhitneyu(group1['fitness_gen_100'], group2['fitness_gen_100'])
        p_value*=bonferroni_correction

        print(f"across encodings for stiff and size changing, gate {gate}, p-value: {p_value}")


# across different encodings, size or stiffness 
for gate in ["AND", "NAND", "OR", "XOR"]:
        pair_1 = [False, True]
        pair_2 = [True, False]
        group1 = df[(df["gate"] == gate) & (df["encoding"] == "size") & (df["stiffness"] == pair_1[0]) & (df["size"] == pair_1[1])] 
        group2 = df[(df["gate"] == gate) & (df["encoding"] == "stiffness") & (df["stiffness"] == pair_2[0]) & (df["size"] == pair_2[1])]

        stat, p_value = mannwhitneyu(group1['fitness_gen_100'], group2['fitness_gen_100'])
        p_value*=bonferroni_correction

        print(f"across encodings for stiff or size changing, gate {gate}, p-value: {p_value}")