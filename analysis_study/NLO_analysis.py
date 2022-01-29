import pandas as pd

#В каком году отмечается наибольшее количество случаев наблюдения НЛО в США?
def new_func():
    df = pd.read_csv('data/NLO.csv', sep=',')
    return df

df = new_func()
df['Time'] = pd.to_datetime(df.Time)
print(df['Time'].dt.year.mode()[0])

#Найдите средний интервал времени (в днях) между двумя последовательными случаями наблюдения НЛО в штате Невада (NV).
df['Date'] = df['Time'].dt.date
print(df[df['State']=='NV']['Date'].diff().dt.days.mean())
