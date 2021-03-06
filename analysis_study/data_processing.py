from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sber_data = pd.read_csv('data/sber_data.csv')
rint(sber_data.head())
print(sber_data.tail())
print(sber_data.shape)
print(sber_data ['sub_area'].nunique())
print(sber_data ['price_doc'].max())

# Проверим, влияет ли уровень экологической обстановки в районе на цену квартиры. Постройте коробчатую 
# диаграмму цен на квартиры (price_doc) в зависимости от уровня экологической обстановки в районе (ecology). 
# Какой уровень ценится на рынке меньше всего?


fig = plt.figure(figsize=(10, 7))
boxplot = sns.boxplot(
    data=sber_data,
    x='price_doc',
    y='ecology',
)


# Постройте диаграмму рассеяния, которая покажет, как цена на квартиру (price_doc) связана 
# с расстоянием до центра Москвы (kremlin_km). Выберите все верные утверждения.

fig = plt.figure(figsize=(10, 7))
boxplot = sns.jointplot(
    data=sber_data,
    y='price_doc',
    x='kremlin_km',
)


# В библиотеке pandas специально для этого реализован метод isnull(). Этот метод возвращает новый DataFrame, 
# в ячейках которого стоят булевы значения True и False. True ставится на месте, где ранее находилось значение NaN.
# Посмотрим на результат работы метода на нашей таблице:

print((sber_data.isnull().tail()))

# Первый способ — это вывести на экран названия столбцов, где число пропусков больше 0. 
# Для этого вычислим средний по столбцам результат метода isnull(). Получим долю пропусков в каждом столбце.
# Умножаем на 100 %, находим столбцы, где доля пропусков больше 0, сортируем по убыванию и выводим результат:

cols_null_percent = sber_data.isnull().mean() * 100
cols_with_null = cols_null_percent[cols_null_percent>0].sort_values(ascending=False)
print(cols_with_null)

# Итак, можно увидеть, что у нас большое число пропусков (более 47 %) в столбце hospital_beds_raion 
# (количество больничных коек в округе). 
# Далее у нас идут столбцы с числом пропусков чуть больше 20 %: 
# preschool_quota (число мест в детском саду в районе)
# school_quota (число мест в школах в районе);
# life_sq (жилая площадь здания в квадратных метрах). 
# 
# Менее одного процента пропусков содержат признаки:
# floor (число этажей в доме)
# metro_min_walk (время от дома до ближайшего метро пешком в минутах);
# metro_km_walk (расстояние до ближайшего метро в километрах);
# railroad_station_walk_km (расстояние до ближайшей ж. д. станции в километрах);
# railroad_station_walk_min (время до ближайшей ж. д. станции пешком в минутах). 

# Эти соотношения дают базовое понимание, какие дальнейшие преобразования со столбцами предстоит производить. 
# Например, уже сейчас ясно, что столбец, в котором почти половина данных пропущена, 
# не может дать нам полезной информации при прогнозировании. Если мы попытаемся его как-то исправить, 
# мы можем только навредить и «нафантазировать» лишнего, поэтому от него, возможно, стоит избавиться. 
# А вот столбцы с менее 1 % пропусков легко можно скорректировать: заполнить отсутствующие значения какими-то числами.

# Иногда столбцов с пропусками становится слишком много и прочитать информацию о них из списка признаков 
# с цифрами становится слишком затруднительно — цифры начинают сливаться воедино. 
# Можно воспользоваться столбчатой диаграммой, чтобы визуально оценить соотношение 
# числа пропусков к числу записей. Самый быстрый способ построить её — использовать метод plot():

cols_with_null.plot(
    kind='bar',
    figsize=(10, 4),
    title='Распределение пропусков в данных'
)

# Ещё один распространённый способ визуализации пропусков — тепловая карта. 
# Её часто используют, когда столбцов с пропусками не так много (меньше 10). 
# Она позволяет понять не только соотношение пропусков в данных, но и их характерное местоположение в таблице. 
# Для создания такой тепловой карты можно воспользоваться результатом метода isnull(). 
# Ячейки таблицы, в которых есть пропуск, будем отмечать жёлтым цветом, а остальные — синим. 
# Для этого создадим собственную палитру цветов тепловой карты с помощью метода color_pallete() из библиотеки seaborn

colors = ['blue', 'yellow'] 
fig = plt.figure(figsize=(10, 4))
cols = cols_with_null.index
ax = sns.heatmap(
    sber_data[cols].isnull(),
    cmap=sns.color_palette(colors),
)

# На полученной тепловой карте мы не видим чётких процентных соотношений для числа пропусков в данных, 
# однако мы можем увидеть места их концентрации в таблице. Например, видно, 
# что признаки preschool_quota и school_quota очень сильно связаны друг с другом по части пропусков:
# во всех записях, где хотя бы один не определён, не указан и второй 
# (жёлтые линии для двух этих признаков полностью совпадают друг с другом).


# Предварительно создадим копию исходной таблицы — drop_data, чтобы не повредить её. 
# Зададимся порогом в 70 %: будем оставлять только те столбцы, в которых 70 и более 
# процентов записей не являются пустыми . После этого удалим записи, в которых содержится хотя бы один пропуск. 
# Наконец, выведем информацию о числе пропусков и наслаждаемся нулями. 

#создаем копию исходной таблицы
drop_data = sber_data.copy()
#задаем минимальный порог: вычисляем 70% от числа строк
thresh = drop_data.shape[0]*0.7
#удаляем столбцы, в которых более 30% (100-70) пропусков
drop_data = drop_data.dropna(how='any', thresh=thresh, axis=1)
#удаляем записи, в которых есть хотя бы 1 пропуск
drop_data = drop_data.dropna(how='any', axis=0)
#отображаем результирующую долю пропусков
drop_data.isnull().mean()

print(drop_data.shape)

# Вся сложность заключается в выборе метода заполнения. Важным фактором при выборе метода является
# распределение признаков с пропусками. Давайте выведем их на экран. 
# В pandas это можно сделать с помощью метода hist():

cols = cols_with_null.index
sber_data[cols].hist(figsize=(20, 8))

# Итак, рассмотрим несколько рекомендаций.
# 
# Для распределений, похожих на логнормальное, где пик близ нуля, а далее наблюдается постепенный 
# спад частоты, высока вероятность наличия выбросов (о них мы поговорим чуть позже). 
# Математически доказывается, что среднее очень чувствительно к выбросам, а вот медиана — нет. 
# Поэтому предпочтительнее использовать медианное значение для таких признаков.
# 
# Если признак числовой и дискретный (например, число этажей, школьная квота), то их заполнение средним/медианой
# является ошибочным, так как может получиться число, которое не может являться значением этого признака. 
# Например, количество этажей — целочисленный признак, а расчёт среднего может дать 2.871. 
# Поэтому такой признак заполняют либо модой, либо округляют до целого числа (или нужного 
# количества знаков после запятой) среднее/медиану.
# 
# Категориальные признаки заполняются либо модальным значением, либо, если вы хотите оставить информацию о 
# пропуске в данных, значением 'unknown'. На наше счастье, пропусков в категориях у нас нет.
# 
# Иногда в данных бывает такой признак, основываясь на котором, можно заполнить пропуски в другом. 
# Например, в наших данных есть признак full_sq (общая площадь квартиры). 
# Давайте исходить из предположения, что, если жилая площадь (life_sq) неизвестна, 
# то она будет равна суммарной площади!


# Заполнение значений осуществляется с помощью метода fillna(). Главный параметр метода — value 
# (значение, на которое происходит заполнение данных в столбце). Если метод вызывается от имени всего DataFrame, 
# то в качестве value можно использовать словарь, где ключи — названия столбцов таблицы, 
# а значения словаря — заполняющие константы. 
# Создадим такой словарь, соблюдая рекомендации, приведённые выше, а также копию исходной таблицы. 
# Произведём операцию заполнения с помощью метода fillna() и удостоверимся, что пропусков в данных больше нет:

#создаем копию исходной таблицы
fill_data = sber_data.copy()
#создаем словарь имя столбца: число(признак) на который надо заменить пропуски
values = {
    'life_sq': fill_data['full_sq'],
    'metro_min_walk': fill_data['metro_min_walk'].median(),
    'metro_km_walk': fill_data['metro_km_walk'].median(),
    'railroad_station_walk_km': fill_data['railroad_station_walk_km'].median(),
    'railroad_station_walk_min': fill_data['railroad_station_walk_min'].median(),
    'hospital_beds_raion': fill_data['hospital_beds_raion'].mode()[0],
    'preschool_quota': fill_data['preschool_quota'].mode()[0],
    'school_quota': fill_data['school_quota'].mode()[0],
    'floor': fill_data['floor'].mode()[0]
}
#заполняем пропуски в соответствии с заявленным словарем
fill_data = fill_data.fillna(values)
#выводим результирующую долю пропусков
fill_data.isnull().mean()

# Посмотрим, на то, как изменились распределения наших признаков:

cols = cols_with_null.index
fill_data[cols].hist(figsize=(20, 8))

# Обратите внимание на то, как сильно изменилось распределение для признака hospital_beds_raion.
# Это связано с тем, что мы заполнили модальным значением почти 47 % общих данных. 
# В результате мы кардинально исказили исходное распределение признака, что может плохо сказаться на модели.

# Посмотрим на реализацию. Как обычно, создадим копию indicator_data исходной таблицы.
# В цикле пройдёмся по столбцам с пропусками и будем добавлять в таблицу новый признак 
# (с припиской "was_null"), который получается из исходного с помощью применения метода isnull(). 
# После чего произведём обычное заполнение пропусков, которое мы совершали ранее, 
# и выведем на экран число отсутствующих значений в столбце, чтобы убедиться в результате:

#создаем копию исходной таблицы
indicator_data = sber_data.copy()
#в цикле пробегаемся по названиям столбцов с пропусками
for col in cols_with_null.index:
    #создаем новый признак-индикатор как col_was_null
    indicator_data[col + '_was_null'] = indicator_data[col].isnull()
#создаем словарь имя столбца: число(признак) на который надо заменить пропуски   
values = {
    'life_sq': indicator_data['full_sq'],
    'metro_min_walk': indicator_data['metro_min_walk'].median(),
    'metro_km_walk': indicator_data['metro_km_walk'].median(),
    'railroad_station_walk_km': indicator_data['railroad_station_walk_km'].median(),
    'railroad_station_walk_min': indicator_data['railroad_station_walk_min'].median(),
    'hospital_beds_raion': indicator_data['hospital_beds_raion'].mode()[0],
    'preschool_quota': indicator_data['preschool_quota'].mode()[0],
    'school_quota': indicator_data['school_quota'].mode()[0],
    'floor': indicator_data['floor'].mode()[0]
}
#заполняем пропуски в соответствии с заявленным словарем
indicator_data = indicator_data.fillna(values)
#выводим результирующую долю пропусков
indicator_data.isnull().mean()

# Метод исходит из предположения, что, если дать модели информацию о том, что в ячейке ранее была пустота, 
# то она будет меньше доверять таким записям и меньше учитывать её в процессе обучения. 
# Иногда такие фишки действительно работают, иногда не дают эффекта, а иногда и вовсе могут ухудшить 
# результат обучения и затруднить процесс обучения.

# Наверняка вы уже догадались, что необязательно использовать один метод. 
# Вы можете их комбинировать. Например, мы можем:
# удалить столбцы, в которых более 30 % пропусков;
# удалить записи, в которых более двух пропусков одновременно;
# заполнить оставшиеся ячейки константами.
# 
# Посмотрим на реализацию такого подхода в коде:

#создаём копию исходной таблицы
combine_data = sber_data.copy()

#отбрасываем столбцы с числом пропусков более 30% (100-70)
n = combine_data.shape[0] #число строк в таблице
thresh = n*0.7
combine_data = combine_data.dropna(how='any', thresh=thresh, axis=1)

#отбрасываем строки с числом пропусков более 2 в строке
m = combine_data.shape[1] #число признаков после удаления столбцов
combine_data = combine_data.dropna(how='any', thresh=m-2, axis=0)

#создаём словарь 'имя_столбца': число (признак), на который надо заменить пропуски 
values = {
    'life_sq': combine_data['full_sq'],
    'metro_min_walk': combine_data['metro_min_walk'].median(),
    'metro_km_walk': combine_data['metro_km_walk'].median(),
    'railroad_station_walk_km': combine_data['railroad_station_walk_km'].median(),
    'railroad_station_walk_min': combine_data['railroad_station_walk_min'].median(),
    'preschool_quota': combine_data['preschool_quota'].mode()[0],
    'school_quota': combine_data['school_quota'].mode()[0],
    'floor': combine_data['floor'].mode()[0]
}
#заполняем оставшиеся записи константами в соответствии со словарем values
combine_data = combine_data.fillna(values)
#выводим результирующую долю пропусков
print(combine_data.isnull().mean())



# Пусть у нас есть признак, по которому мы будем искать выбросы. Давайте рассчитаем его 
# статистические показатели (минимум, максимум, среднее, квантили) и по ним попробуем определить наличие аномалий.
# Сделать это можно с помощью уже знакомого вам метода describe(). Рассчитаем статистические показатели 
# для признака жилой площади (life_sq)

print(sber_data['life_sq'].describe())


# Что нам говорит метод describe()? Во-первых, у нас есть квартиры с нулевой жилой площадью.
# Во-вторых, в то время как 75-й квантиль равен 43, максимум превышает 7 тысяч квадратных метров 
# (целый дворец, а не квартира!). 
# Найдём число квартир с нулевой жилой площадью:

print(sber_data[sber_data['life_sq'] == 0].shape[0])

# А теперь выведем здания с жилой площадью более 7 000 квадратных метров

print(sber_data[sber_data['life_sq'] > 7000])

# Вот он, красавец! Выброс налицо: гигантская жилая площадь (life_sq), да ещё почти в 
# 100 раз превышает общую площадь (full_sq).
# Логичен вопрос: а много ли у нас таких квартир, у которых жилая площадь больше, чем суммарная?
# Давайте проверим это с помощью фильтрации:

outliers = sber_data[sber_data['life_sq'] > sber_data['full_sq']]
print(outliers.shape[0])

# Таких квартир оказывается 37 штук. Подобные наблюдения уже не поддаются здравому смыслу — 
# они являются ошибочными, и от них стоит избавиться. Для этого можно воспользоваться методом drop() 
# и удалить записи по их индексам:

cleaned = sber_data.drop(outliers.index, axis=0)
print(f'Результирующее число записей: {cleaned.shape[0]}')
 
 
## Результирующее число записей: 30434
# Ещё пример: давайте посмотрим на признак числа этажей (floor).

print(sber_data['floor'].describe())

# Снова видим подозрительную максимальную отметку в 77 этажей. 
# Проверим все квартиры, которые находятся выше 50 этажей:

print(sber_data[sber_data['floor']> 50])

# Построим гистограмму и коробчатую диаграмму для признака полной площади (full_sq):

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
histplot = sns.histplot(data=sber_data, x='full_sq', ax=axes[0]);
histplot.set_title('Full Square Distribution');
boxplot = sns.boxplot(data=sber_data, x='full_sq', ax=axes[1]);
boxplot.set_title('Full Square Boxplot')

# В соответствии с этим алгоритмом напишем функцию outliers_iqr(), которая вам может 
# ещё не раз пригодиться в реальных задачах. Эта функция принимает на вход DataFrame 
# и признак, по которому ищутся выбросы, а затем возвращает потенциальные выбросы, 
# найденные с помощью метода Тьюки, и очищенный от них датасет.
# Квантили вычисляются с помощью метода quantile(). Потенциальные выбросы 
# определяются при помощи фильтрации данных по условию выхода за пределы верхней или нижней границы.

def outliers_iqr(data, feature):
    x = data[feature]
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    outliers = data[(x<lower_bound) | (x > upper_bound)]
    cleaned = data[(x>lower_bound) & (x < upper_bound)]
    return outliers, cleaned

# Применим эту функцию к таблице sber_data и признаку full_sq, а также выведем размерности результатов:

outliers, cleaned = outliers_iqr(sber_data, 'full_sq')
print(f'Число выбросов по методу Тьюки: {outliers.shape[0]}')
print(f'Результирующее число записей: {cleaned.shape[0]}')


# Число выбросов по методу Тьюки: 963
# Результирующее число записей: 29508

# Согласно классическому методу Тьюки, под выбросы у нас попали 963 записи в таблице.
# Давайте построим гистограмму и коробчатую диаграмму на новых данных cleaned_sber_data:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
histplot = sns.histplot(data=cleaned, x='full_sq', ax=axes[0])
histplot.set_title('Cleaned Full Square Distribution')
boxplot = sns.boxplot(data=cleaned, x='full_sq', ax=axes[1])
boxplot.set_title('Cleaned Full Square Boxplot')

# Давайте немного модифицируем функцию outliers_iqr(). Добавьте в неё параметры left и right, 
# которые задают число IQR влево и вправо от границ ящика (пусть по умолчанию они равны 1.5).
# Функция, как и раньше, должна возвращать потенциальные выбросы и очищенный DataFrame.

def outliers_iqr_mod(data, feature, left=1.5, right=1.5):
    x = data[feature]
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)
    outliers = data[(x<lower_bound) | (x> upper_bound)]
    cleaned = data[(x>lower_bound) & (x < upper_bound)]
    return outliers, cleaned

# Давайте ослабим границы метода Тьюки справа и усилим их влево. Примените модифицированную функцию 
# outliers_iqr_mod() к признаку full_sq из таблицы sber_data данным с параметрами left=1 и right=6.
# Результаты работы поместите в переменные outliers и cleaned. 
# Чему равно результирующее число выбросов в данных?

outliers, cleaned = outliers_iqr_mod(sber_data, 'full_sq', left=1, right=6)
print(f'Число выбросов по методу Тьюки: {outliers.shape[0]}')
print(f'Результирующее число записей: {cleaned.shape[0]}')

# Если мы построим гистограмму и коробчатую диаграмму на полученных данных, то увидим вот такую картинку:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
histplot = sns.histplot(data=cleaned, x='full_sq', ax=axes[0])
histplot.set_title('Cleaned Full Square Distribution')
boxplot = sns.boxplot(data=cleaned, x='full_sq', ax=axes[1])
boxplot.set_title('Cleaned Full Square Boxplot')

# Согласитесь, это уже больше похоже на реальный рынок недвижимости: основная часть 
# квартир имеет площадь в интервале от 25 до 85 кв. м, а далее частота наблюдений постепенно падает.


# Построим две гистограммы признака расстояния до МКАД (mkad_km): 
# первая — в обычном масштабе, а вторая — в логарифмическом. 
# Логарифмировать будем с помощью функции log() из библиотеки numpy 
# (натуральный логарифм — логарифм по основанию числа e). Признак имеет среди своих значений 0. 
# Из математики известно, что логарифма от 0 не существует, 
# поэтому мы прибавляем к нашему признаку 1, чтобы не логарифмировать нули и не получать предупреждения.

fig, axes = plt.subplots(1, 2, figsize=(15, 4))

#гистограмма исходного признака
histplot = sns.histplot(sber_data['mkad_km'], bins=30, ax=axes[0])
histplot.set_title('MKAD Km Distribution');

#гистограмма в логарифмическом масштабе
log_mkad_km= np.log(sber_data['mkad_km'] + 1)
histplot = sns.histplot(log_mkad_km , bins=30, ax=axes[1])
histplot.set_title('Log MKAD Km Distribution')

# Левое распределение напоминает логнормальное распределение с наличием потенциальных выбросов-«пеньков», 
# далеко отстоящих от основной массы наблюдений.
# Взяв натуральный логарифм от левого распределения, мы получаем правое, которое напоминает 
# слегка перекошенное нормальное. Слева от моды (самого высокого столбика) наблюдается чуть
# больше наблюдений, нежели справа. По-научному это будет звучать так: «распределение имеет левостороннюю асимметрию».
# Примечание: Численный показатель асимметрии можно вычислить с помощью метода skew()

print(log_mkad_km.skew())

# Асимметрия распределения называется правосторонней, если она положительная:
# Асимметрия распределения называется левосторонней, если она отрицательная:

# Давайте реализуем алгоритм метода z-отклонения. Описание алгоритма метода:
# вычислить математическое ожидание  (среднее) и стандартное отклонение  признака ;
# вычислить нижнюю и верхнюю границу интервала как:
#  lower_bound = mu - 3 * sigma
# upper_bound = mu + 3 * sigma
# найти наблюдения, которые выходят за пределы границ.
# Напишем функцию outliers_z_score(), которая реализует этот алгоритм. 
# На вход она принимает DataFrame и признак, по которому ищутся выбросы. 
# В дополнение добавим в функцию возможность работы в логарифмическом масштабе: 
# для этого введём аргумент log_scale. Если он равен True, то будем логарифмировать 
# рассматриваемый признак, иначе — оставляем его в исходном виде.
# Как и раньше, функция будет возвращать выбросы и очищенные от них данные:

def outliers_z_score(data, feature, log_scale=False):
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - 3 * sigma
    upper_bound = mu + 3 * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned

# Применим эту функцию к таблице sber_data и признаку mkad_km, а также выведем размерности результатов:

outliers, cleaned = outliers_z_score(sber_data, 'mkad_km', log_scale=True)
print(f'Число выбросов по методу z-отклонения: {outliers.shape[0]}')
print(f'Результирующее число записей: {cleaned.shape[0]}')
# Число выбросов по методу z-отклонения: 33
# Результирующее число записей: 30438

# Итак, метод z-отклонения нашел нам 33 потенциальных выброса по признаку расстояния до МКАД.
# Давайте узнаем, в каких районах (sub_area) представлены эти квартиры:

print(outliers['sub_area'].unique())


# Возможно, мы не учли того факта, что наш логарифм распределения всё-таки не идеально нормален 
# и в нём присутствует некоторая асимметрия. Возможно, стоит дать некоторое «послабление» 
# на границы интервалов? Давайте отдельно построим гистограмму прологарифмированного распределения, 
# а также отобразим на гистограмме вертикальные линии, соответствующие среднему 
# (центру интервала в методе трёх сигм) и границы интервала . Вертикальные линии можно построить с помощью 
# метода axvline(). Для среднего линия будет обычной, а для границ интервала — пунктирной (параметр ls ='--'):

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
log_mkad_km = np.log(sber_data['mkad_km'] + 1)
histplot = sns.histplot(log_mkad_km, bins=30, ax=ax)
histplot.axvline(log_mkad_km.mean(), color='k', lw=2)
histplot.axvline(log_mkad_km.mean()+ 3 * log_mkad_km.std(), color='k', ls='--', lw=2)
histplot.axvline(log_mkad_km.mean()- 3 * log_mkad_km.std(), color='k', ls='--', lw=2)
histplot.set_title('Log MKAD Km Distribution')

# Итак, что мы графически построили интервал метода трёх сигм поверх нашего распределения. 
# Он показывает, какие наблюдения мы берем в интервал, а какие считаем выбросами. Легко заметить, 
# среднее значение (жирная вертикальная линия) находится левее моды, это свойство распределений с 
# левосторонней асимметрией. Также видны наблюдения, которые мы не захватили своим интервалом 
# (небольшой пенек правее верхней границы) — это и есть наши квартиры из из поселений "Роговское" и "Киевский".
# Очевидно, что если немного (меньше чем на одну сигму) "сдвинуть" верхнюю границу вправо, 
# мы захватим эти наблюдения. Давайте сделаем это?!

# авайте расширим правило трёх сигм, чтобы иметь возможность особенности данных. 
# Добавьте в функцию outliers_z_score() параметры left и right, которые будут задавать число сигм 
# (стандартных отклонений) влево и вправо соответственно, определяющее границы метода z-отклонения.
# По умолчанию оба параметры равны 3. Результирующую функцию назовите outliers_z_score_mod().

def outliers_z_score_mod(data, feature, left = 3, right = 3, log_scale = False):
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliners = data[(x < lower_bound) | (x > upper_bound)]
    cleand = data[(x > lower_bound) & (x < upper_bound)]
    return outliners, cleand


# Проверьте, что будет, если дать «послабление» вправо, увеличив число сигм. Наша задача — 
# узнать, начиная с какой границы поселения «Роговское» и «Киевское» перестают считаться выбросами.
# Примените свою функцию outliers_z_score_mod() к признаку mkad_km с параметрами 
# left=3, right=3.5, log_scale=True. Чему равно результирующее число выбросов?

outliers, cleaned = outliers_z_score_mod(sber_data, 'mkad_km', right=3.5, log_scale=True)
print(f'Число выбросов по методу z-отклонения: {outliers.shape[0]}')
print(f'Результирующее число записей: {cleaned.shape[0]}')


# Постройте гистограмму для признака price_doc в логарифмическом масштабе. 
# А также, добавьте на график линии, отображающие среднее и границы интервала для метода трех сигм. 
# Выберите верные утверждения

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
log_price = np.log(sber_data['price_doc'])
histplot = sns.histplot(log_price, bins=30, ax=ax)
histplot.set_title('Log Price Distribution');
histplot.axvline(log_price.mean(), color='k', lw=2)
histplot.axvline(log_price.mean()+ 3 * log_price.std(), color='k', ls='--', lw=2)
histplot.axvline(log_price.mean()- 3 * log_price.std(), color='k', ls='--', lw=2);


# Найдите потенциальные выбросы с помощью метода z-отклонения. 
# Используйте логарифмический масштаб распределения. Сделайте «послабление» на 
# 0.7 сигм в в обе стороны распределения. Сколько выбросов вы получили?

outliers, cleaned = outliers_z_score_mod(sber_data, 'price_doc', left=3.7, right=3.7, log_scale=True)
print(f'Число выбросов по методу z-отклонения: {outliers.shape[0]}')

# Добавьте фишку с логарифмированием в свою функцию outliers_iqr_mod(). Д
# обавьте в неё параметр log_scale. Если он выставлен в True, то производится логарифмирование 
# признака. Примените полученную функцию к признаку price_doc. Число межквартильных размахов в 
# обе стороны обозначьте как 3. Чему равно число выбросов, полученных таким методом?
# При логарифмировании признака price_doc добавлять к нему 1 не нужно, он не имеет нулевых значений!

def outliers_iqr_mod(data, feature, left=1.5, right=1.5, log_scale=False):
    if log_scale:
        x = np.log(data[feature])
    else:
        x= data[feature]
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)
    outliers = data[(x<lower_bound) | (x > upper_bound)]
    cleaned = data[(x>lower_bound) & (x < upper_bound)]
    return outliers, cleaned
outliers, cleaned = outliers_iqr_mod(sber_data, 'price_doc', left=3, right=3, log_scale=True)
print(f'Число выбросов по методу Тьюки: {outliers.shape[0]}')


# Способ обнаружения дубликатов зависит от того, что именно вы считаете дубликатом. 
# Например, за дубликаты можно посчитать записи, у которых совпадают все признаки или их часть. 
# Если в таблице есть столбец с уникальным идентификатором (id), вы можете попробовать поискать дубликаты по нему:
# одинаковые записи могут иметь одинаковый id.
# Проверим, есть у нас такие записи: для этого сравним число уникальных значений в столбце id 
# с числом строк. Число уникальных значений вычислим с помощью метода nunique():

sber_data['id'].nunique() == sber_data.shape[0]

# Найдём число полных дубликатов таблице sber_data. Предварительно создадим список столбцов dupl_columns, 
# по которым будем искать совпадения (все столбцы, не включая id). 
# Создадим маску дубликатов с помощью метода duplicated() и произведём фильтрацию. 
# Результат заносим в переменную sber_duplicates. Выведем число строк в результирующем DataFrame:

dupl_columns = list(sber_data.columns)
dupl_columns.remove('id')

mask = sber_data.duplicated(subset=dupl_columns)
sber_duplicates = sber_data[mask]
print(f'Число найденных дубликатов: {sber_duplicates.shape[0]}')