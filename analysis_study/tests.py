
import pandas as pd


countries_df = pd.DataFrame({
    'country': ['Англия', 'Канада', 'США', 'Россия', 'Украина', 'Беларусь', 'Казахстан'],
    'population': [56.29, 38.05, 322.28, 146.24, 45.5, 9.5, 17.04],
    'square': [133396, 9984670, 9826630, 17125191, 603628, 207600, 2724902]
})
countries_df['density'] = countries_df['population'] / countries_df['square'] * 1e6
print(round(countries_df['density'].mean(), 2))


import seaborn as sns
print(sns.__version__)

import plotly
import plotly.express as px
print(plotly.__version__)


def get_rate(rate):
    rate_list = ['грн', 'USD', 'EUR', 'белруб', 'KGS', 'сум', 'AZN', 'KZT', 'руб.']
    r_list = rate.split(' ')
    rate_type = rate_list[-1]
    if rate_type in rate_list :
        rate_type =  r_list[-2]
    return rate_type

 data_hh ['Курс ЗП']= data_hh['ЗП'].apply(get_rate)
 
 
  

fig = px.scatter(
   data_hh,
   x = 'Опыт работы (год)',
   y = 'Возраст»',
   trendline_options = {0:0, 100:100},
   trendline_scope = 'trace'
)

fig.show()


# Ваша задача — очистить данную таблицу от пропусков следующим образом:
# Если признак имеет больше 50 % пропущенных значений, удалите его.
# Для оставшихся данных: если в строке более двух пропусков, удалите строку.
# Для оставшихся данных: числовые признаки заполните средним значением, а категориальные — модой.
# У вас должна получиться следующая таблица df:


df = pd.read_csv('./Root/data/test_data.csv')
thresh = df.shape[0]*0.5
df = df.dropna(thresh=thresh, axis=1)
thresh2 = df.shape[1] - 2
df = df.dropna(thresh=thresh2, axis=0)
df = df.fillna({
    'one': df['one'].mean(),
    'two': df['two'].mean(),
    'four': df['four'].mode()[0]})

null_data = data_hh.isnull().sum()
print(null_data[null_data > 0])

data_hh_drop = data_hh['Последнее/нынешнее место работы'].dropna(how='any')
data_hh_drop = data_hh_drop['Последняя/нынешняя должность'].dropna(how='any')

values = {
    'Опыт работы (месяц)': data_hh_drop['Опыт работы (месяц)'].median()
    }

data_hh_new = data_hh_drop.fillna(values)


duplicates = data_hh[data_hh.duplicated(subset=data_hh.columns)]
data = data_hh.drop_duplicates()
print(duplicates.shape[0])