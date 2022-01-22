import pandas as pd

melb_data = pd.read_csv('data/melb_data.csv', sep=',')

print(melb_data.loc[15, 'Price'])
print(melb_data.loc[90, 'Date'])

print(round(melb_data.loc[3521, 'Landsize'] / melb_data.loc[1690, 'Landsize']))

print(melb_data.head())

print(melb_data.tail(7))

print(melb_data.shape)

print(melb_data.info())

melb_data['Car'] = melb_data['Car'].astype('int64')
melb_data['Bedroom'] = melb_data['Bedroom'].astype('int64')
melb_data['Bathroom'] = melb_data['Bathroom'].astype('int64')
melb_data['Propertycount'] = melb_data['Propertycount'].astype('int64')
melb_data['YearBuilt'] = melb_data['YearBuilt'].astype('int64')
melb_data.info()

print(melb_data.describe().loc[:, ['Distance', 'BuildingArea' , 'Price']])

print(melb_data.describe(include=['object']))

print(melb_data['Regionname'].value_counts())

print(melb_data['Regionname'].value_counts(normalize=True))

print(melb_data['Type'].value_counts(normalize=True))

print(melb_data['Price'].mean()) # средняя цена объекта недвижимости

print(melb_data['Car'].max()) # максимальное число парковочных мест

# А теперь представим, что риэлторская ставка для всех компаний за продажу недвижимости составляет 12%.
# Найдём общую прибыльность риэлторского бизнеса в Мельбурне. Результат округлим до сотых:
rate = 0.12
income = melb_data['Price'].sum() * rate
print('Total income of real estate agencies:', round(income, 2))


#Найдём, насколько медианная площадь территории отличается от её среднего значения.
# Вычислим модуль разницы между медианой и средним и разделим результат на среднее, 
# чтобы получить отклонение в долях:
landsize_median = melb_data['Landsize'].median() 
landsize_mean =  melb_data['Landsize'].mean()
print(abs(landsize_median - landsize_mean)/landsize_mean)

# Вычислим, какое число комнат чаще всего представлено на рынке недвижимости
print(melb_data['Rooms'].mode())

# наиболее распространённое название района
print(melb_data['Regionname'].mode())

# максимальное количество домов на продажу в районе
print(melb_data['Propertycount'].max())

# стандартное отклонение от центра города до недвижимости
print(round(melb_data['Distance'].std()))

# отклонение медианного значения площади здания от его среднего значения
building_median = melb_data['BuildingArea'].median() 
building_mean =  melb_data['BuildingArea'].mean()
deviance = abs(building_median - building_mean)/building_mean
print(round(deviance * 100, 2))

# сколько спален чаще всего встречается в Мельбурне
print(melb_data['Bedroom'].mode())

# создаем маску для фильтрации:
mask = melb_data['Price'] > 2000000
print(mask)

# Для фильтрации нужно просто подставить переменную mask в индексацию DataFrame. 
# Маска показывает, какие строки нужно оставлять в результирующем наборе, а какие — убирать 
# (выведем первые пять строк отфильтрованной таблицы)
print(melb_data[mask].head())

# Также вовсе не обязательно заносить маску в отдельную переменную — 
# можно сразу вставлять условие в операцию индексации DataFrame, например
print(melb_data[melb_data['Price'] > 2000000])

# Найдём количество зданий с тремя комнатами.
# Для этого отфильтруем таблицу по условию: обратимся к результирующей таблице по столбцу 
# Rooms и найдём число строк в ней с помощью атрибута shape
print(melb_data[melb_data['Rooms'] == 3].shape[0])

# Усложним прошлый пример и найдём число трёхкомнатных домов с ценой менее 300 тысяч:
print(melb_data[(melb_data['Rooms'] == 3) & (melb_data['Price'] < 300000)].shape[0])


# Таких зданий оказалось всего три. 
# Немного «ослабим» условие: теперь нас будут интересовать дома с ценой менее 300 тысяч, 
# у которых либо число комнат равно 3 либо площадь домов более 100 квадратных метров
print(melb_data[((melb_data['Rooms'] == 3) | (melb_data['BuildingArea'] > 100)) & (melb_data['Price'] < 300000)].shape[0])

# Фильтрацию часто сочетают со статистическими методами. 
# Давайте найдём максимальное количество комнат в таунхаусах. 
# Так как в результате фильтрации получается DataFrame, то обратимся к нему по столбцу Rooms и найдём максимальное значение
print(melb_data[melb_data['Type'] == 't']['Rooms'].max())

#найдём медианную площадь здания у объектов, чья цена выше средней.
# Для того чтобы оградить наш код от нагромождений, предварительно создадим переменную со средней ценой:
mean_price = melb_data['Price'].mean()
print(melb_data[melb_data['Price'] > mean_price]['BuildingArea'].median())

# у скольких объектов недвижимости отсутствууют ванные комнаты
print(melb_data[melb_data['Bathroom'] == 0].shape[0])

# сколько объектов недвижимости проданных Нильсеном и стоимостью более 3 млн
print(melb_data[(melb_data['SellerG'] == 'Nelson') & (melb_data['Price'] > 3e6)].shape[0])

# какая минимальная стоимость участка без здания
print(melb_data[(melb_data['BuildingArea'] == 0)]['Price'].min())

print(round(melb_data[(melb_data['Price']<1e6) & ((melb_data['Rooms']>5) | (melb_data['YearBuilt'] > 2015))]['Price'].mean()))

# в каком районе Мельбурна продаются виллы и котеджи
print(melb_data[(melb_data['Type'] == 'h') & (melb_data['Price'] < 3000000)]['Regionname'].mode())
