import pandas as pd

melb_data = pd.read_csv('data/melb_data.csv', sep=',')

print(melb_data.loc[15, 'Price'])
print(melb_data.loc[90, 'Date'])

print(round(melb_data.loc[3521, 'Landsize'] / melb_data.loc[1690, 'Landsize']))