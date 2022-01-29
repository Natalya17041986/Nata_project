# Датасет представляет собой таблицу с информацией о 300 тысячах поездок за первые пять дней сентября 2018 года и включает в 
# себя следующую информацию:
# starttime — время начала поездки (дата, время);
# stoptime — время окончания поездки (дата, время);
# start station id — идентификатор стартовой стоянки;
# start station name — название стартовой стоянки;
# start station latitude, start station longitude — географическая широта и долгота стартовой стоянки;
# end station id — идентификатор конечной стоянки;
# end station name — название конечной стоянки;
# end station latitude, end station longitude — географическая широта и долгота конечной стоянки;
# bikeid — идентификатор велосипеда;
# usertype — тип пользователя (Customer — клиент с подпиской на 24 часа или на три дня, Subscriber — подписчик с годовой арендой велосипеда);
# birth year — год рождения клиента;
# gender — пол клиента (0 — неизвестный, 1 — мужчина, 2 — женщина).

import pandas as pd

data= pd.read_csv('data/citibike-tripdata.csv', sep=',')

# сколько пропусков в столбце start station id:
print(data.info)

# идентификатор самой популярной стартовой площадки:
print(data['start station id'].mode()[0])

# велосипед с каким идентификатором является самым популярным
print(data['bikeid'].mode()[0])

# какой тип клиентов явлется преобладающим
mode_usertype = data['usertype'].mode()[0]
count_mode_user = data[data['usertype'] == mode_usertype].shape[0]
print(round(count_mode_user / data.shape[0], 2))

# кто больше занимается спортом мужчины или женщины
male_count = data[data['gender'] == 1].shape[0]
female_count = data[data['gender'] == 0].shape[0]
print(max([male_count, female_count]))

# удаляем признаки идентификаторов стоянок:
data.drop(['start station id', 'end station id'], axis=1, inplace=True)
print(data.shape[1])

# заменяем признак год рождения на возвраст и считаем сколько поездок совершено клиентами старше 60 лет
data['age'] = 2018 - data['birth year']
data.drop(['birth year'], axis=1, inplace=True)
print(data[data['age'] > 60].shape[0])

# создаем признак длительности поездки и счтаем среднюю длительность поездки в секундах
data['starttime'] = pd.to_datetime(data['starttime'])
data['stoptime'] = pd.to_datetime(data['stoptime'])
data['trip duration'] = (data['stoptime'] - data['starttime']).dt.seconds
print(round(data['trip duration'].mean(), 2)

#Создайте «признак-мигалку» weekend, который равен 1, если поездка начиналась в выходной день (суббота или воскресенье), 
# и 0 — в противном случае. Выясните, сколько поездок начиналось в выходные.
weekday = data['starttime'].dt.dayofweek
data['weekend'] = weekday.apply(lambda x: 1 if x ==5 or x == 6 else 0)
data['weekend'].sum()

#Создайте признак времени суток поездки time_of_day. Время суток будем определять из часа начала поездки. 
# Условимся, что:
# поездка совершается ночью (night), если её час приходится на интервал от 0 (включительно) до 6 (включительно) часов;
# поездка совершается утром (morning), если её час приходится на интервал от 6 (не включительно) до 12 (включительно) часов;
# поездка совершается днём (day), если её час приходится на интервал от 12 (не включительно) до 18 (включительно) часов;
# поездка совершается вечером (evening), если её час приходится на интервал от 18 (не включительно) до 23 часов (включительно).
# Во сколько раз количество поездок, совершённых днём, больше, чем количество поездок, совёршенных ночью, 
# за представленный в данных период времени?
def get_time_of_day(time):
    if 0 <= time <= 6:
        return 'night'
    elif 6 < time <= 12:
        return 'morning'
    elif 12 < time <= 18:
        return 'day'
    elif 18 < time <= 23:
        return 'evening'
    else:
        return 'else'
data['time_of_day'] = data['starttime'].dt.hour.apply(get_time_of_day)
a = data[data['time_of_day'] == 'day'].shape[0]
b = data[data['time_of_day'] == 'night'].shape[0]
print(round(a / b))