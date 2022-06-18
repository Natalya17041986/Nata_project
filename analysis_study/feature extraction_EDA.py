import pandas as pd
import seaborn as sns

wine = pd.read_csv('data/wine_cleared.csv')
print(wine)

# Создадим новый признак price_round, означающий округлённую до целого числа цену за бутылку вина:

# для удобства сразу преобразуем признак в int

wine['price_round'] = wine['price'].round().astype(int)

# Помимо округления для создания новых признаков могут применяться такие популярные операции, 
# как логарифмирование числового признака или извлечение его корня. Это подразумевает создание признаков в 
# наиболее удобной форме для обучения модели.

# Регулярные выражения (regexp, или regex) — это механизм для поиска и замены текста. 
# Это шаблоны, которые используются для поиска соответствующей части текста.
# Например, с помощью такого регулярного выражения [^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+ можно найти любой email 
# в тексте.

# Реализация такого механизма существует в pandas в работе со строками. Для того чтобы найти все числа в 
# каждом значении серии, воспользуемся методом str.findall(). Метод возвращает все совпадения с заданным 
# шаблоном в серии pandas. 
# Выполните код для нахождения года вина при помощи регулярного выражения:

regex = '\d{4}' # регулярное выражение для нахождения чисел
wine['year'] = wine['title'].str.findall(regex).str.get(0)

# Разберём регулярное выражение \d+:
# \d — класс символов, обозначает соответствие цифрам в диапазоне цифр [0-9];
# {4} в шаблоне означает искать четыре вхождения символа, указанного ранее. 
# В нашем случае это будут четырехзначные числа.
# Таким образом, \d{4} означает четырехзначных чисел в заданной строке.

# Однако при поиске числа методом data['title'].str.findall(regex) результатом выполнения является список 
# найденных цифр. Поэтому необходимо извлечь первый элемент из списка найденных методом str.get(0), 
# где 0 — первый элемент в списке найденных чисел.

# В наборе данных винных обзоров самая популярная страна-производитель вина — США. 
# Возможно, это не случайность, и факт производства в США влияет на рейтинг вина. Выделим этот факт.
# Вы можете создать новый бинарный признак is_usa и присвоить ему 1 в случае, если вино произведено в США, иначе — 0.

wine['is_usa'] = wine['country'].apply(lambda x: 1 if x == 'US' else 0)

# В наборе данных также есть ещё две страны, которые являются не менее популярными производителями вина.

import seaborn as sns

import matplotlib.pyplot as plt

# Выберите из списка две самых популярных (помимо США) страны, производящих вино.
print(wine['country'].value_counts())

# Создайте бинарные признаки is_france, is_italy наподобие признака is_usa.

wine['is_italy'] = wine['country'].apply(lambda x: 1 if x == 'Italy' else 0)

wine['is_france'] = wine['country'].apply(lambda x: 1 if x == 'France' else 0)

print(wine['is_france'].sum())
print(wine['is_italy'].sum())

# Попробуем вывести правило в определении качества вина: старые вина лучше молодых.
# В нашем датасете 40 % вин старше 2010 года. 

# преобразуем признак year в объект datetime для удобного сравнения дат
wine['year'] = pd.to_datetime(wine['year'], errors='coerce')

wine['old_wine'] = wine['year'].apply(lambda x: 1 if x.year < 2010 else 0)

print(wine['old_wine'].sum())



# Создайте новый признак locality из признака title, который будет обозначать название долины/местности
# производства вина. Например, в названии вина Rainstorm 2013 Pinot Gris (Willamette Valley) locality 
# будет Willamette Valley. В названии Tandem 2011 Ars In Vitro Tempranillo-Merlot (Navarra) — Navarra.

regex = '\((.*?)\)'
wine['locality'] = wine['title'].str.findall(regex).str.get(0)

# Часто маленькие страны с небольшим количеством населения имеют узкую специализацию. 
# Например, в производстве вина особенно успешны Франция, Италия, Испания, Новая Зеландия. 
# Чтобы проверить, влияет ли на качество вина населённость, выясним информацию о населении страны, 
# в котором была произведена бутылка вина.

import pandas as pd
country_population = pd.read_csv('data/country_population.csv', sep=';')


print(country_population)

print(country_population.loc[country_population['country'] == 'Italy'])

# Далее сопоставим значения из датасета country_population и страной-производителем вина. 
# На основе значений населения из country_population заполним новый признак country_population.
# Используем для этого функцию для объединения датасетов join. Для объединения используем аргумент on='country',
# указывая столбец, по которому объединяем датафреймы:

wine_1 = wine.join(country_population.set_index('country'), on = 'country')


country_area = pd.read_csv('data/country_area.csv', sep=';')
print(country_area)

# Создайте новый признак area_country — площадь страны, аналогичный признаку country_population.
# Какая площадь страны у вина под названием 'Gård 2014 Grand Klasse Reserve Lawrence Vineyards Viognier 
# (Columbia Valley (WA))'? Ответ вводите без пробелов, округлите до целых.

wine_2 = wine_1.join(country_area.set_index('country'), on = 'country')
print(wine_2)

new = wine_2.loc[wine_2['title'] == 'Gård 2014 Grand Klasse Reserve Lawrence Vineyards Viognier (Columbia Valley (WA))']
print(new['area'])



# РАБОТА С ДАТАМИ

# Давайте теперь приступим к практическим заданиям. В следующих заданиях мы будем использовать 
# срез базы данных из колл-центра. Компания хочет предсказывать, какому из клиентов стоит звонить сегодня, 
# а какому — нет.
# Давайте рассмотрим, из каких признаков состоит срез данных:
# client_id — идентификатор клиента в базе;
# agent_date — время соединения с агентом;
# created_at — время соединения с клиентом (начало разговора);
# end_date — время окончания соединения с клиентом (конец разговора).
# 
# Прочитаем данные:

import pandas as pd 

# инициализируем информацию о звонках
calls_list = [
    [460, '2013-12-17 04:55:39', '2013-12-17 04:55:44', '2013-12-17 04:55:45'],
    [12, '2013-12-16 20:03:20', '2013-12-16 20:03:22', '2013-12-16 20:07:13'],
    [56, '2013-12-16 20:03:20', '2013-12-16 20:03:20', '2013-12-16 20:05:04'],
    [980, '2013-12-16 20:03:20','2013-12-16 20:03:27', '2013-12-16 20:03:29'],
    [396, '2013-12-16 20:08:27', '2013-12-16 20:08:28','2013-12-16 20:12:03'],
    [449, '2013-12-16 20:03:20', '2013-12-16 20:03:25','2013-12-16 20:05:00'],
    [397, '2013-12-16 20:08:25', '2013-12-16 20:08:27', '2013-12-16 20:09:59'],
    [398, '2013-12-16 20:01:23', '2013-12-16 20:01:23', '2013-12-16 20:04:58'],
    [452, '2013-12-16 20:03:20', '2013-12-16 20:03:21','2013-12-16 20:04:55'],
    [440, '2013-12-16 20:03:20', '2013-12-16 20:04:26', '2013-12-16 20:04:32']
]

calls = pd.DataFrame(calls_list, columns = ['client_id',  'agent_date', 'created_at' ,'end_date'])

# преобразовываем признаки в формат datetime для удобной работы

calls['agent_date'] = pd.to_datetime(calls['agent_date'])
calls['created_at'] = pd.to_datetime(calls['created_at'])
calls['end_date'] = pd.to_datetime(calls['end_date'])

print(calls)

# Все признаки в наборе данных, за исключением номера клиента, представляют собой дату и время. 
# Давайте создадим несколько признаков из этих данных.
# Мы можем посчитать, сколько примерно длилось время разговора клиента и сотрудника компании — длительность разговора. 
# Подсчитаем разницу между датой и временем начала разговора с клиентом и датой и временем окончания звонка.

calls['duration'] = (calls['end_date'] - calls['created_at']).dt.seconds
print(calls)

# Таким образом мы получили новый признак duration — длительность разговора в секундах.
# Давайте создадим ещё несколько признаков на основе существующих.

# Подсчитайте, сколько секунд тратят сотрудники компании на дозвон клиенту. 
# Результат запишите в новый признак time_connection
calls['time_connection'] = (calls['created_at'] - calls['agent_date']).dt.seconds
print(calls)
print(calls['time_connection'].sum())


# Создайте новый признак is_connection — факт соединения с клиентом. 
# Признак будет равен 1 в случае, если разговор состоялся и продлился больше 10 секунд, иначе — 0

calls['is_connection'] = calls['duration'].apply(lambda x: 1 if x > 10 else 0)

print(calls)

print(calls['is_connection'].sum())

# Создайте признак time_diff — разницу в секундах между началом звонка(не разговора) и его окончанием.

calls['time_diff'] = (calls['end_date'] - calls['agent_date']).dt.seconds

print(calls)

print(calls['time_diff'].sum())

# Итак, мы получили четыре новых признака для нашего набора данных: duration, time_connection, is_connection,
# time_diff. После генерации признаков из дат исходные признаки agent_date, created_at, 
# end_date нам больше не нужны — передать на вход модели мы им не сможем, так как большинство моделей 
# машинного обучения умеют работать только с числами, даты и текст ей недоступны, поэтому удалим их:

calls = calls.drop(columns=['agent_date', 'created_at' ,'end_date'], axis=1)

# Итоговый набор данных включает в себя колонки client_id, duration, time_connection, 
# is_connection, time_diff. После генерации признаков специалисты по машинному обучению проводят 
# отбор признаков. Этому вы научитесь далее в юните про отбор признаков.
# Таким образом, мы получили набор данных с признаками, которые можно подать на вход модели, и не потеряли 
# важную информацию о событиях, произошедших в даты набора данных. 

print(wine_2.info())
print(wine_2['year'])

# Создайте признак количество дней с момента произведения вина — years_diff для датасета винных обзоров. 
# За дату отсчёта возьмите 12 января 2022 года. В ответ впишите максимальное количество дней 
# с момента произведения вина. Ответ округлите до целого числа.

wine_2['year'] = pd.to_datetime(wine_2['year'])
wine_2['years_diff'] = (pd.to_datetime("2022-01-12") - wine_2['year']).dt.days
print(wine_2['years_diff'].max())

##  Кодирование признаков. Методы

# Ниже мы рассмотрим методы кодирования, обозначенные в блок-схеме. Для кодирования категориальных 
# признаков мы будем использовать библиотеку category_encoders. Это удобная библиотека 
# для кодирования категориальных переменных различными методами.

import category_encoders as ce

# Рассмотрим следующие популярные способы кодирования: 
# порядковое кодирование (Ordinal Encoding);
# однократное кодирование (OneHot Encoding); 
# бинарное кодирование (Binary Encoding).
# 
# Создадим обучающий набор для кодирования порядковых признаков — ассортимент небольшого магазина с одеждой, 
# где size — буквенное обозначение размера одежды, type — тип изделия.

import pandas as pd
# инициализируем информацию об одежде
clothing_list = [
    ['xxs', 'dress'],
    ['xxs', 'skirt'],
    ['xs', 'dress'],
    ['s', 'skirt'],
    ['m', 'dress'],
    ['l', 'shirt'],
    ['s', 'coat'],
    ['m', 'coat'],
    ['xxl', 'shirt'],
    ['l', 'dress']
]

clothing = pd.DataFrame(clothing_list, columns = ['size',  'type'])
print(clothing)

# Выполним теперь кодирование порядкового признака size и type признака в Python. Порядковое кодирование в 
# библиотеке реализовано в классе OrdinalEncoder. По умолчанию все строковые столбцы будут закодированы.
# Метод fit_transform устанавливает соответствия для кодирования и преобразовывает данные в соответствие с ними. 
# Затем используем метод concat() для добавления закодированного признака в датафрейм data.

ord_encoder = ce.OrdinalEncoder()
data_bin = ord_encoder.fit_transform(clothing[['size', 'type']])
clothing = pd.concat([clothing, data_bin], axis=1)

print(clothing)

# Порядковое кодирование может успешно использоваться для кодирования порядковых признаков.
# Мы можем закодировать признак size — размер одежды со значениями xxs, xs, s соответственно в значения 1, 2, 3. 
# Это будет логично, и моделью не будут сделаны выводы о неправильном порядке. Увеличение размера будет 
# соответствовать логическому увеличению кода этого значения: xxs меньше xs, и числовой код 1 (xxs) меньше, 
# чем числовой код 2 (xs).
# Однако порядковое кодирование плохо работает для номинальных признаков. Ошибку при кодировании мы не получим, 
# но алгоритмы машинного обучения не могут различать категории и числовые признаки, поэтому могут быть 
# сделаны выводы о неправильном порядке. 


# Используйте ранее изученные методы кодирования и закодируйте признак year в датасете винных 
# обзоров порядковым кодированием.

ord_encoder = ce.OrdinalEncoder(cols=['year'])

year_col = ord_encoder.fit_transform(wine_2['year'])
wine_2 = pd.concat([wine_2, year_col], axis=1)



# Закодируем признак type однократным кодированием. Закодируем признак type в Python. Используем класс 
# OneHotEncoding библиотеки category_encoders. Укажем в cols наименование признака type для кодировки, 
# иначе будут закодированы все строковые столбцы.

clothing_list = [
    ['xxs', 'dress'],
    ['xxs', 'skirt'],
    ['xs', 'dress'],
    ['s', 'skirt'],
    ['m', 'dress'],
    ['l', 'shirt'],
    ['s', 'coat'],
    ['m', 'coat'],
    ['xxl', 'shirt'],
    ['l', 'dress']
]

clothing = pd.DataFrame(clothing_list, columns = ['size',  'type'])

import category_encoders as ce
encoder = ce.OneHotEncoder(cols=['type']) # указываем столбец для кодирования
type_bin = encoder.fit_transform(clothing['type'])
clothing = pd.concat([clothing, type_bin], axis=1)

print(clothing)

# На самом деле метод однократного кодирования реализован в pandas в функции pd.get_dummies(). 
# Для выполнения кодирования достаточно передать в функцию DataFrame и указать столбцы, для которых 
# должно выполняться кодирование. По умолчанию кодирование выполняется для всех столбцов типа object:

clothing_dummies = pd.get_dummies(clothing, columns=['type'])

print(clothing)
print(clothing_dummies)

# Новые бинарные признаки также часто называются dummy-признаками или dummy-переменными.  


# В нашем наборе данных винных обзоров признак, обозначающий имя сомелье (taster_name), является номинальным.
# Закодируйте его, используя One-Hot Encoding.

import category_encoders as ce
encoder = ce.OneHotEncoder(cols=['taster_name']) # указываем столбец для кодирования
type_bin = encoder.fit_transform(wine_2['taster_name'])
wine_2 = pd.concat([wine_2, type_bin], axis=1)

print(wine_2)

# Закодируем бинарным способом признак type в Python. Используем класс BinaryEncoder библиотеки category_encoders.

import category_encoders as ce # импорт для работы с кодировщиком
bin_encoder = ce.BinaryEncoder(cols=['type']) # указываем столбец для кодирования
type_bin = bin_encoder.fit_transform(clothing['type'])
clothing = pd.concat([clothing, type_bin], axis=1)

print(clothing)

# Результатом кодирования будет три новых признака: type_0, type_1, typе_2: 


# Вернёмся к нашему примеру с винным датасетом. Признак country содержит много уникальных значений — используем двоичную кодировку признака.
import category_encoders as ce # импорт для работы с кодировщиком
bin_encoder = ce.BinaryEncoder(cols=['country']) # указываем столбец для кодирования
type_bin = bin_encoder.fit_transform(wine_2['country'])
wine_2 = pd.concat([wine_2, type_bin], axis=1)


print(wine_2)


# На основе изученного материала определите подходящий способ кодирования признака
# taster_twitter_handle из датасета винных обзоров и закодируйте его.
# Признак taster_twitter_handle номинальный и имеет много уникальных значений (более 15). 
# Следует выбрать binary encoder.

import category_encoders as ce # импорт для работы с кодировщиком
bin_encoder = ce.BinaryEncoder(cols=['taster_twitter_handle']) # указываем столбец для кодирования
type_bin = bin_encoder.fit_transform(wine_2['taster_twitter_handle'])
wine_2 = pd.concat([wine_2, type_bin], axis=1)

print(wine_2)


list_of_dicts = [
 {'product': 'Product1', 'price': 1200, 'payment_type': 'Mastercard'},
 {'product': 'Product2', 'price': 3600, 'payment_type': 'Visa'},
 {'product': 'Product3', 'price': 7500, 'payment_type': 'Amex'}
]
df = pd.DataFrame(list_of_dicts)

print(df)

import category_encoders as ce
encoder = ce.OneHotEncoder(cols=['product','payment_type']) # указываем столбец для кодирования
type_bin = encoder.fit_transform(df[['product','payment_type']])
df = pd.concat([df, type_bin], axis=1)

print(df)

## НОРМАЛИЗАЦИЯ

# Используем библиотеку numpy для создания массивов случайных чисел различных распределений. 
# Выполните этот код, чтобы создать обучающий набор различных распределений:

import numpy as np 
import pandas as pd

np.random.seed(34)

# для нормализации, стандартизации
from sklearn import preprocessing

# Для графиков
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib inline
matplotlib.style.use('ggplot')

# сгенерируем датасет из случайных чисел
df = pd.DataFrame({ 
    # Бета распределение, 5 – значение альфа, 1 – значение бета, 1000 – размер
    'beta': np.random.beta(5, 1, 1000) * 60,
    
    # Экспоненциальное распределение, 10 – "резкость" экспоненты, 1000 – размер
    'exponential': np.random.exponential(10, 1000),
    
    # Нормальное распределение, 10 – среднее значение р., 2 – стандартное отклонение, 1000 – количество сэмплов
    'normal_p': np.random.normal(10, 2, 1000),
    
    # Нормальное распределение, 10 – среднее значение р., 10 – стандартное отклонение, 1000 – количество сэмплов
    'normal_l': np.random.normal(10, 10, 1000),
})

# Копируем названия столбцов, которые теряются при использовании fit_transform()
col_names = list(df.columns)

print(df)

# Сгенерированные распределения выбраны случайным образом, однако вы можете встретить их, например, в 
# таких наборах данных:
# Бета-распределение моделирует вероятность. Например, коэффициент конверсии клиентов, купивших что-то на сайте.
# Экспоненциальное распределение, предсказывающее периоды времени между событиями. Например, время ожидания автобуса.
# Нормальное распределение, например распределение роста и веса человека.
# 
# Рассмотрим распределения на графике. Метод визуализации kdeplot() — это метод визуализации распределения 
# наблюдений в наборе данных. Он представляет собой непрерывную линию плотности вероятности. Подробнее об этой 
# функции вы можете прочитать в руководстве.

# зададим параметры холста, название и визуализируем кривые распределения:
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('Исходные распределения')

# kdeplot() (KDE – оценка плотности ядра) – специальный метод для графиков распределений
sns.kdeplot(df['beta'], ax=ax1, label ='beta')
sns.kdeplot(df['exponential'], ax=ax1, label ='exponential')
sns.kdeplot(df['normal_p'], ax=ax1, label ='normal_p')
sns.kdeplot(df['normal_l'], ax=ax1, label ='normal_l')
plt.legend()

plt.show()

# Признаки распределены по-разному: смещены влево, вправо, присутствуют отрицательные величины. 
# Попробуем нормализовать их.
# Зафиксируем описательные статистики до преобразований.

print(df.describe()) # описательные характеристики

# Для нормализации данных мы будем использовать уже знакомую нам библиотеку sklearn.

# Класс MinMaxScaler делает вышеописанную нормализацию автоматически при помощи функции преобразования 
# fit_transform. Вы познакомитесь с ней подробнее в модулях машинного обучения.

# инициализируем нормализатор MinMaxScaler
mm_scaler = preprocessing.MinMaxScaler()

# копируем исходный датасет
df_mm = mm_scaler.fit_transform(df)

# Преобразуем промежуточный датасет в полноценный датафрейм для визуализации
df_mm = pd.DataFrame(df_mm, columns=col_names)

fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('После нормализации MinMaxScaler')

sns.kdeplot(df_mm['beta'], ax=ax1)
sns.kdeplot(df_mm['exponential'], ax=ax1)
sns.kdeplot(df_mm['normal_p'], ax=ax1)
sns.kdeplot(df_mm['normal_l'], ax=ax1)

plt.show()

print(df_mm.describe()) # описательные характеристики

# Проведём нормализацию распределений признаков из обучающего примера, используя класс RobustScaler.

# инициализируем нормализатор RobustScaler
r_scaler = preprocessing.RobustScaler()

# копируем исходный датасет
df_r = r_scaler.fit_transform(df)

# Преобразуем промежуточный датасет в полноценный датафрейм для визуализации
df_r = pd.DataFrame(df_r, columns=col_names)

fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('Распределения после RobustScaler')

sns.kdeplot(df_r['beta'], ax=ax1)
sns.kdeplot(df_r['exponential'], ax=ax1)
sns.kdeplot(df_r['normal_p'], ax=ax1)
sns.kdeplot(df_r['normal_l'], ax=ax1)

plt.show()

print(df_r.describe()) # описательные характеристики

# Из описательных статистик видно, что RobustScaler не масштабирует данные в заданный интервал,
# как делает это MinMaxScaler. Однако распределения не сохранили своё исходное состояние. 
# Левый хвост экспоненциального распределения стал практически незаметным. То же произошло и с 
# бета-распределением. Они стали более нормальными.

## СТАНДАРТИЗАЦИЯ

# Чтобы понять, как стандартизация меняет распределение, рассмотрим метод стандартизации StandardScaler в Python.

# Для стандартизации используем класс StandardScaler.

# инициализируем стандартизатор StandardScaler
s_scaler = preprocessing.StandardScaler()

# копируем исходный датасет
df_s = s_scaler.fit_transform(df)

# Преобразуем промежуточный датасет в полноценный датафрейм для визуализации
df_s = pd.DataFrame(df_s, columns=col_names)

fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('Распределения после StandardScaler')

sns.kdeplot(df_s['beta'], ax=ax1)
sns.kdeplot(df_s['exponential'], ax=ax1)
sns.kdeplot(df_s['normal_p'], ax=ax1)
sns.kdeplot(df_s['normal_l'], ax=ax1)

plt.show()


print(df_s.describe()) # описательные характеристики

# Нормализуйте признак price. Используйте подходящий тип нормализации.
# В ответе напишите результат выполнения кода data['price'].sum(), округлённый до целого.

r_scaler = preprocessing.RobustScaler()
wine_3 = r_scaler.fit_transform(wine_2[['price']]) # окпируем исходный дата сет
wine_3 = pd.DataFrame(wine_3, columns = ['price']) # преобразуем промежуточный дата сет в полноценный датафрейм для визуализации

print(wine_3['price'].sum())

# Стандартизируйте исходный признак price.
# В ответе напишите результат выполнения кода data['price'][129968]. Ответ округлите до сотых.

# инициализируем стандартизатор StandardScaler
s_scaler = preprocessing.StandardScaler()

# копируем исходный датасет
wine_4 = s_scaler.fit_transform(wine_2[['price']])
wine_4 = pd.DataFrame(wine_4, columns = ['price']) # преобразуем промежуточный дата сет в полноценный датафрейм для визуализации

print(wine_4['price'][129968])

## МУЛЬТИКОЛЛИНЕАРНОСТЬ

# Процесс корреляционного анализа и удаление сильно скоррелированных признаков относят к одному из 
# методов отбора признаков.
# Рассмотрим отбор признаков в Python. Для этого воспользуемся обучающим датасетом о цветках ириса.
# Данные содержат 150 экземпляров ириса, по 50 экземпляров трех видов — Ирис щетинистый (Iris setosa), 
# Ирис виргинский (Iris virginica) и Ирис разноцветный (Iris versicolor). Для каждого экземпляра 
# измерялись четыре характеристики (в сантиметрах):
# sepal length — длина наружной доли околоцветника;
# sepal width — ширина наружной доли околоцветника;
# petal length — длина внутренней доли околоцветника;
# petal width — ширина внутренней доли околоцветника.

# На основании этого набора данных требуется построить модель, определяющую вид растения по данным измерений. 
# Прочитаем датасет и посмотрим на первые несколько строк.

import pandas as pd

iris = pd.read_csv('data/iris.csv')
print(iris.head())

# Будем исследовать признаки, которые могут влиять на variety — sepal.length, sepal.width, petal.length, petal.width.
# Проведём корреляционный анализ датасета и используем для этого тепловую карту корреляций признаков.

import seaborn as sns # импортируем seaborn для построения графиков
sns.heatmap(iris.corr(), annot=True) # включаем отображение коэффициентов
plt.show()

#Получаем следующую тепловую карту для признаков:

# Как мы выяснили из тепловой карты корреляций, у нас есть три пары сильно скоррелированных признаков:
# sepal.length и petal.width, petal.length и sepal.length, petal.width и petal.length.
# Начнём с самого высокого коэффициента корреляции в паре признаков: petal.width и petal.length 0,96. Удалим любой признак из этой пары, например petal.width, так как он коррелирует ещё и с признаком sepal.length:

iris = iris.drop(['petal.width'], axis=1)

# Однако второй признак petal.length из этой пары также сильно коррелирует с признаком sepal.length. Удалим и его:

iris = iris.drop(['petal.length'], axis=1)

# Посмотрим на результат:

print(iris.head())

#  нас осталось всего два признака с коэффициентом корреляции -0.12: sepal.width и sepal.length, и 
# признак, который необходимо предсказать — variety. Связь между оставшимися признаками очень слабая, 
# поэтому эти признаки будут включены в итоговый набор данных для обучения.

# Это означает, что всего два признака — sepal length (длина наружной доли околоцветника) и sepal width 
# (ширина наружной доли околоцветника) сообщают модели то же самое, что и исходный набор признаков. 
# Мы уменьшили количество признаков, не потеряв при этом информацию о данных. А признаки petal length 
# (длина внутренней доли околоцветника) и petal width (ширина внутренней доли околоцветник) сообщают лишнюю 
# информацию для модели. Значит, они не нужны для построения модели.

# Таким образом, анализ мультиколлинеарности и исключение сильно скоррелированных признаков 
# помогает отобрать признаки для модели, уменьшить количество признаков, исключить повторяющуюся информацию.


# Проведите корреляционный анализ всего набора данных и отберите только необходимые признаки для 
# предсказания рейтинга вина.
# Удалять признак рейтинг — points нельзя!
# Для простоты вычислений можете использовать только корреляцию Пирсона.

print(wine_2.corr())

# Построим график корреляции всех величин. Для простоты воспользуемся корреляцией Пирсона.

# работа с визуализацией
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(wine_2.corr(), annot=True, linewidths=.5, ax=ax)
plt.show()

# Удалим самые сильно скоррелированные пары
wine_2 = wine_2.drop(['is_usa', 'is_france', 'is_italy', 'price_round', 'area'], axis=1)

# Проверяем, что сильно скоррелированных признаков не осталось
sns.heatmap(wine_2.corr(), annot=True, linewidths=.5, ax=ax)
plt.show()


## ЗАДАНИЯ НА ПРОВЕРКУ

# Датасет болезней сердца содержит информацию о пациентах и переменную предсказания target — наличие у пациента
# болезни сердца.
# Датасет содержит следующие признаки:
# age — возраст
# sex — пол (1 - мужчина, 0 - женщина)
# cp — тип боли в груди (4 значения)
# trestbps — артериальное давление в покое
# chol — холестерин сыворотки в мг/дл
# fbs — уровень сахара в крови натощак > 120 мг/дл
# restecg — результаты электрокардиографии в покое (значения 0,1,2)
# thalach — достигнута максимальная частота сердечных сокращений
# exang — стенокардия, вызванная физической нагрузкой
# oldpeak — депрессия ST, вызванная физической нагрузкой, по сравнению с состоянием покоя
# slope — наклон пикового сегмента ST при нагрузке
# ca — количество крупных сосудов (0-3), окрашенных при флюроскопии
# thal — дефект, где 3 = нормальный; 6 = фиксированный дефект; 7 = обратимый дефект

# Создайте новый признак old, где 1 — при возрасте пациента более 60 лет.
# В ответ введите результат выполнения кода heart['old'].sum().

heart = pd.read_csv('data/heart.csv')

heart['old'] = heart['age'].apply(lambda x: 1 if x > 60 else 0)

print(heart['old'].sum())

# Создайте новый признак trestbps_mean, который будет обозначать норму давления в среднем для его возраста и пола. 
# trestbps — систолическое артериальное давление в состоянии покоя.
# Информацию о среднем давлении для возраста и пола возьмите из этой таблицы. В таблице систолическое давление 
# написано первым, перед дробной чертой.

def get_trestbps_mean(sex, age):
    pressure = [
        [116, 120, 127, 137, 144, 159],
        [123, 126, 129, 135, 142, 142]
    ]

    if age < 21:
        return pressure[int(sex)][0]
    elif age >= 61:
        return pressure[int(sex)][5]
    else:
        return pressure[int(sex)][int((age - 1) // 10 - 1)]


heart['trestbps_mean'] = heart.apply(lambda row: get_trestbps_mean(row['sex'], row['age']), axis=1)

print(heart['trestbps_mean'] [300])

# Проанализируйте датасет и выберите категориальные признаки.
print(heart.info())

import category_encoders as ce
encoder = ce.OneHotEncoder(cols=['cp','restecg','slope','ca','thal']) # указываем столбец для кодирования
type_bin = encoder.fit_transform(heart[['cp','restecg','slope','ca','thal']])
heart = pd.concat([heart, type_bin], axis=1)


print(heart.info())

# Нормализуйте все числовые признаки подходящим способом.
# В ответе напишите стандартное отклонение признака chol. Ответ округлите до шести знаков после запятой.


r_scaler = preprocessing.RobustScaler()
heart_2 = r_scaler.fit_transform(heart[['age','sex','cp','trestbps', 'chol', 'fbs','restecg','thalach','exang','oldpeak',
                                      'slope','ca','thal','target','old','trestbps_mean','cp_1', 
                                      'cp_2', 'cp_3', 'cp_4', 'restecg_1', 'restecg_2', 'restecg_3',
                                      'slope_1', 'slope_2', 'slope_3', 'ca_1', 'ca_2','ca_3','ca_4', 'ca_5',
                                     'thal_1','thal_2','thal_3','thal_4']]) # окпируем исходный дата сет
heart_2 = pd.DataFrame(heart_2, columns = [['age','sex','cp','trestbps', 'chol', 'fbs','restecg','thalach','exang','oldpeak',
                                      'slope','ca','thal','target','old','trestbps_mean','cp_1', 
                                      'cp_2', 'cp_3', 'cp_4', 'restecg_1', 'restecg_2', 'restecg_3',
                                      'slope_1', 'slope_2', 'slope_3', 'ca_1', 'ca_2','ca_3','ca_4', 'ca_5',
                                     'thal_1','thal_2','thal_3','thal_4']]) # преобразуем промежуточный дата сет в полноценный датафрейм для визуализации


print(heart_2.describe())

# Проведите корреляционный анализ и отберите признаки для будущей модели. Выберите пары сильно скоррелированных признаков.

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(heart_2.corr(), annot=True, linewidths=.5, ax=ax)
plt.show()