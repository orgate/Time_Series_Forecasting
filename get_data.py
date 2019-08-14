import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import sys

#ps = pd.read_csv("prices-split-adjusted.csv", sep = ",")
#ps["date"] = pd.to_datetime(ps.date).dt.date
#ps_csv_export = ps.to_csv(r'prices_split_adjusted_modified_date.csv',index=False)
ps = pd.read_csv("prices_split_adjusted_modified_date.csv", sep = ",")


#fig, ax = plt.subplots(figsize=(10,10))
#sns.heatmap(ps_stk.corr(), ax=ax, annot=True, linewidths=0.05, fmt='.2f', cmap="magma")
#plt.show()
#ps["date"] = pd.to_datetime(ps.date).dt.date

ps_pvt = pd.pivot_table(ps,index=["date"],values=["open","close","low","high"],
               columns=["symbol"],fill_value=0)

#ps_pvt.sort("date")

Nsymbols = len(set(ps.symbol))
Ndates = len(ps_pvt.index)

ps_open_t = [ps_pvt.loc[ps_pvt.index[i],'open'] for i in range(Ndates)]
ps_close_t = [ps_pvt.loc[ps_pvt.index[i],'close'] for i in range(Ndates)]
ps_low_t = [ps_pvt.loc[ps_pvt.index[i],'low'] for i in range(Ndates)]
ps_high_t = [ps_pvt.loc[ps_pvt.index[i],'high'] for i in range(Ndates)]


ps_open_t_arr = np.array(ps_open_t).T