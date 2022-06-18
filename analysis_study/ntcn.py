
import pandas as pd
import seaborn as sns


import sweetviz as sv

import dtale
data = pd.read_csv('data/wine.csv')
d = dtale.show(data)
d
