# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:39:14 2019
exploratory data analysis
@author: Moorthy
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt

bcdata= pd.read_csv("E:\\NCI\\Sem2\\ADM\\project\\data\\breast-cancer-wisconsin.data")

bcdata=bcdata.replace('?',np.NaN)
bcdata['Bare_Nuclei'] = bcdata['Bare_Nuclei'].fillna(method='ffill')
bcdata.drop(['Sample_code_number'], 1, inplace=True)

sns.set(style='white', color_codes=True)
plt.figure(figsize=(14, 12))
sns.heatmap(bcdata.astype(float).corr(), linewidths=0.1, square=True, linecolor='white', annot=True)
plt.show()
# 0.91 correlation between Uniformity_of_Cell_Size and Uniformity_of_Cell_Shape




