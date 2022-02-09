import pandas as pd

covid_data = pd.read_csv('data/covid_data.csv')
print(covid_data.head())

# Мы будем работать со следующими столбцами:
# date — дата наблюдения
# province/state — наименование провинции/штата;
# country — наименование страны;
# confirmed — общее число зафиксированных случаев на указанный день;
# deaths — общее число зафиксированных смертей на указанный день;
# recovered — общее число выздоровлений на указанный день.

vaccinations_data = pd.read_csv('data/country_vaccinations.csv')
vaccinations_data = vaccinations_data[
    ['country', 'date', 'total_vaccinations', 
     'people_vaccinated', 'people_vaccinated_per_hundred',
     'people_fully_vaccinated', 'people_fully_vaccinated_per_hundred',
     'daily_vaccinations', 'vaccines']
]

#Данная таблица содержит следующие столбцы:
# country — наименование страны;
# date — дата наблюдения;
# total_vaccinations — общее число введённых вакцин в стране на указанный день;
# people_vaccinated — общее число привитых первым компонентом в стране на указанный день;
# people_vaccinated_per_hundred — процент привитых первым компонентом в стране на указанный день (рассчитывается как );
# people_fully_vaccinated — общее число привитых вторым компонентом в стране на указанный день (первый компонент уже был введён им ранее);
# people_fully_vaccinated_per_hundred — процент привитых вторым компонентом в стране на указанный день (рассчитывается как );
# daily_vaccination — ежедневная вакцинация (число вакцинированных в указанный день);
# vaccines — комбинации вакцин, используемые в стране.

# В таблице covid_data:
# 1 Группируем таблицу по дате и названию страны и рассчитываем суммарные показатели по всем регионам. Тем самым переходим от данных по регионам к данным по странам:

covid_data = covid_data.groupby(
    ['date', 'country'], 
    as_index=False
)[['confirmed', 'deaths', 'recovered']].sum()

# 2 Преобразуем даты в формат datetime с помощью функции pd.to_datetime():

covid_data['date'] = pd.to_datetime(covid_data['date'])

# 3 Создадим признак больных на данный момент (active). Для этого вычтем из общего числа зафиксированных случаев число смертей и число выздоровевших пациентов:

covid_data['active'] = covid_data['confirmed'] - covid_data['deaths'] - covid_data['recovered']
# 4 Создадим признак ежедневного прироста числа заболевших, умерших и выздоровевших людей. Для этого отсортируем данные по названиям стран, а затем по датам. После этого произведём группировку по странам и рассчитаем разницу между «вчера и сегодня» с помощью метода diff():

covid_data = covid_data.sort_values(by=['country', 'date'])
covid_data['daily_confirmed'] = covid_data.groupby('country')['confirmed'].diff()
covid_data['daily_deaths'] = covid_data.groupby('country')['deaths'].diff()
covid_data['daily_recovered'] = covid_data.groupby('country')['recovered'].diff()

# В таблице vaccinations_data достаточно будет преобразовать столбцы в формат datetime:

vaccinations_data['date'] = pd.to_datetime(vaccinations_data['date'])

# За какой период представлены данные в таблице covid_data? 
# В качестве ответа введите даты в формате datetime (без указания времени).

print(f"Данные о заболеваемости предоставлены от {covid_data['date'].min()} до  {covid_data['date'].max()}")

#За какой период представлены данные в таблице vaccinations_data? 
# В качестве ответа введите даты в формате datetime без указания времени.

print(f"Данные о вакцинации предоставлены от {vaccinations_data['date'].min()} до  {vaccinations_data['date'].max()}")

# С помощью метода merge() объедините таблицы covid_data и vaccinations_data по столбцам date и country.
# Тип объединения выставьте так, чтобы в результирующую таблицу попали только наблюдения за период, 
# вычисленный в задании 3.1. То есть в результирующую таблицу должны попасть все записи из таблицы covid_data 
# и из её пересечения с vaccinations_data, но не более. Результат объединения занесите в переменную covid_df.
# Сколько строк и столбцов в таблице covid_df?
covid_df = covid_data.merge(vaccinations_data, on=['date', 'country'], how='left')
print('Число строк: ', covid_df.shape[0])
print('Число столбцов: ', covid_df.shape[1])

# В получившейся в задании 3.3 таблице covid_df создайте признаки death_rate — 
# общий процент смертей среди зафиксированных случаев (летальность) и recover_rate — 
# общий процент случаев выздоровления. Данные характеристики рассчитайте как отношение числа смертей (deaths) 
# и числа выздоровлений (recovered) к числу зафиксированных случаев (confirmed) и умножьте результаты на 100%.

# Какова максимальная летальность в США (United States) за весь период? Ответ округлите до второго знака после запятой.

covid_df['death_rate'] = covid_df['deaths'] / covid_df['confirmed'] * 100
covid_df['recover_rate'] = covid_df['recovered'] / covid_df['confirmed'] * 100
print(round(covid_df[covid_df['country'] == 'United States']['death_rate'].max(), 2))

# Чему равен средний процент выздоровевших в России (Russia)? Ответ округлите до второго знака после запятой

print(round(covid_df[covid_df['country'] == 'Russia']['recover_rate'].mean(), 2))

# Начнём с исследования заболеваемости коронавирусом во всём мире. Для этого первым делом отобразим, 
# как менялось ежедневное число заболевших (daily_confirmed) во всём мире во времени. Далее сгруппируем таблицу 
# по датам и подсчитаем суммарное число зафиксированных случаев по дням. Теперь мы наконец можем построить график 
# с помощью метода plot(). Будем использовать линейный график размером 12x4 
# (попробуйте взять другие числа, чтобы увидеть разницу). Подпишем график и отобразим сетку. Параметр lw (line width) 
# отвечает за ширину линии для линейного графика.

# для отображения графика нужно выгрузить модуль:
import matplotlib.pyplot as plt 
 
grouped_cases = covid_df.groupby('date')['daily_confirmed'].sum()
grouped_cases.plot(
    kind='line',
    figsize=(12, 4),
    title='Ежедневная заболеваемость по всем странам',
    grid = True,
    lw=3
)
# команда для вывода графика как картинки
plt.show()