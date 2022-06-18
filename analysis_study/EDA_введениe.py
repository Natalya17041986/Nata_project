# После просмотра документального фильма о сомелье вы захотели создать прогностическую модель для оценки вин вслепую, 
# как это делает сомелье.
# Определив бизнес-задачу, вы перешли к сбору данных для обучения модели. После нескольких недель парсинга 
# сайта WineEnthusiast вам удалось собрать около 130 тысяч строк обзоров вин для анализа и обучения.
# Вот какие признаки вам удалось собрать:
# country — страна-производитель вина.
# description — подробное описание.
# designation — название виноградника, где выращивают виноград для вина
# points — баллы, которыми WineEnthusiast оценил вино по шкале от 1 до 100.
# price — стоимость бутылки вина.
# province — провинция или штат.
# region_1 — винодельческий район в провинции или штате (например Напа).
# region_2 — конкретный регион. Иногда в пределах винодельческой зоны указываются более конкретные регионы 
# (например Резерфорд в долине Напа), но это значение может быть пустым.
# taster_name — имя сомелье.
# taster_twitter_handle — твиттер сомелье.
# title — название вина, которое часто содержит год и другую подробную информацию.
# variety — сорт винограда, из которого изготовлено вино (например Пино Нуар).
# winery — винодельня, которая производила вино.

import pandas as pd
import seaborn as sns

data = pd.read_csv('data/wine.csv')

# Сколько всего дегустаторов приняло участие в винных обзорах?
print(data.info())

print(data['taster_name'].nunique()) # поиск количества сомелье, которые оценивали вино на сайте

print(data['price'].max()) # поиск максимальной цены на вино

# поиск и удаление дубликатов

mask = data.duplicated()
data_duplicates = data[mask]
print(f'Число найденных дубликатов: {data_duplicates.shape[0]}')

data_dedupped = data.drop_duplicates()

# поиск нулевых значений

cols_null_percent = data_dedupped.isnull().mean() * 100
cols_with_null = cols_null_percent[cols_null_percent>0].sort_values(ascending=False)
print(cols_with_null)

# удаление пустых записей

data_dedupped = data_dedupped.drop(['region_2'], axis=1)

# обрабатываем пропуски в категориальных признаках самым простым вариантом, замена на unknown

data_dedupped ['designation'] = data_dedupped ['designation'].fillna('unknown')
data_dedupped ['region_1'] = data_dedupped ['region_1'].fillna('unknown')
data_dedupped ['taster_name'] = data_dedupped ['taster_name'].fillna('unknown')
data_dedupped ['taster_twitter_handle'] = data_dedupped ['taster_twitter_handle'].fillna('unknown')

# признаки с маленьким количеством пропусков заменим на самые частовречающиеся значения
data_dedupped ['country'] = data_dedupped ['country'].fillna('US')
data_dedupped ['price'] = data_dedupped ['price'].fillna(data_dedupped['price'].mean())
data_dedupped ['province'] = data_dedupped ['province'].fillna('California')
data_dedupped ['variety'] = data_dedupped ['variety'].fillna('Pinot Noir')

# в числовом признаке выберем метод замены средним значением
data_dedupped ['price'] = data_dedupped ['price'].fillna(data_dedupped ['price'].mean())

sns.heatmap(data_dedupped .isnull()) # убеждаемся, что датасет без пропусков

data_dedupped = data_dedupped.to_csv('data/wine_cleared.csv', index=False) # сохраняем очищенный датасет для дальнейшей работы

