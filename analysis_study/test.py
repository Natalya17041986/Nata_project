from sklearn.metrics import matthews_corrcoef


x = [+1, -1, +1, +1] # список значений признака х
y = [+1, +1, +1, -1] # список значений признака y

print(matthews_corrcoef(x, y)) # рассчитаем коэффициент корреляции Мэтьюса




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns

covid_data = pd.read_csv('data/covid_data.csv')
print(covid_data.head())

fig = plt.figure(figsize=(10, 7))
boxplot = sns.boxplot(
    data=croped_covid_df,
    y='country',
    x='death_rate',
    orient='h',
    width=0.9
)
boxplot.set_title('Распределение летальности по странам')
boxplot.set_xlabel('Летальность')
boxplot.set_ylabel('Страна')
boxplot.grid()