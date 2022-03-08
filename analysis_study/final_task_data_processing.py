from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

diabetes_data = pd.read_csv('data/diabetes_data.csv')
print(diabetes_data.head())

# Признаки данных:
# Pregnancies — количество беременностей.
# Glucose — концентрация глюкозы в плазме через два часа при пероральном тесте на толерантность к глюкозе.
# BloodPressure — диастолическое артериальное давление (мм рт. ст.).
# SkinThickness — толщина кожной складки трицепса (мм).
# Insulin — двухчасовой сывороточный инсулин (ме Ед/мл).
# BMI — индекс массы тела
# DiabetesPedigreeFunction — функция родословной диабета (чем она выше, тем выше шанс наследственной заболеваемости).
# Age — возраст.
# Outcome — наличие диабета (0 — нет, 1 — да).

# Начнём с поиска дубликатов в данных. Найдите все повторяющиеся строки в данных и удалите их. 
# Для поиска используйте все признаки в данных. Сколько записей осталось в данных?

duplicates = diabetes_data[diabetes_data.duplicated()]
print('Число дубликтов: {}'.format(duplicates.shape[0]))
diabetes_data_drop = diabetes_data.drop_duplicates()
print('Результирующее число записей: {}'.format(diabetes_data_drop.shape[0]))

# Далее найдите все неинформативные признаки в данных и избавьтесь от них. В качестве порога информативности 
# возьмите 0.99: удалите все признаки, для которых 99 % значений повторяются или 99 % записей уникальны. В ответ запишите имена признаков, которые вы нашли (без кавычек)

#список неинформативных признаков
low_information_cols = [] 

#цикл по всем столбцам
for col in diabetes_data_drop.columns:
    #наибольшая относительная частота в признаке
    top_freq = diabetes_data_drop[col].value_counts(normalize=True).max()
    #доля уникальных значений от размера признака
    nunique_ratio = diabetes_data_drop[col].nunique() / diabetes_data_drop[col].count()
    # сравниваем наибольшую частоту с порогом
    if top_freq > 0.99:
        low_information_cols.append(col)
        print(f'{col}: {round(top_freq*100, 2)}% одинаковых значений')
    # сравниваем долю уникальных значений с порогом
    if nunique_ratio > 0.99:
        low_information_cols.append(col)
        print(f'{col}: {round(nunique_ratio*100, 2)}% уникальных значений')
        

# Попробуйте найти пропуски в данных с помощью метода insull().
# Спойлер: ничего не найдёте. А они есть! Просто они скрыты от наших глаз. В таблице пропуски в столбцах 
# Glucose, BloodPressure, SkinThickness, Insulin и BMI обозначены нулём, поэтому традиционные методы поиска 
# пропусков ничего вам не покажут. Давайте это исправим!
# Замените все записи, равные 0, в столбцах Glucose, BloodPressure, SkinThickness, Insulin и BMI на 
# символ пропуска. Его вы можете взять из библиотеки numpy: np.nan.
# Какая доля пропусков содержится в столбце Insulin? Ответ округлите до сотых.

def nan_function(x):
    return np.nan if x == 0 else x

diabetes_data_drop["Glucose"] = diabetes_data_drop["Glucose"].apply(nan_function)
diabetes_data_drop["BloodPressure"] = diabetes_data_drop["BloodPressure"].apply(nan_function)
diabetes_data_drop["SkinThickness"] = diabetes_data_drop["SkinThickness"].apply(nan_function)
diabetes_data_drop["Insulin"] = diabetes_data_drop["Insulin"].apply(nan_function)
diabetes_data_drop["BMI"] = diabetes_data_drop["BMI"].apply(nan_function)

print(diabetes_data_drop.isnull().mean().round(2).sort_values(ascending=False))

# Удалите из данных признаки, где число пропусков составляет более 30 %. 
# Сколько признаков осталось в ваших данных (с учетом удаленных неинформативных признаков в задании 8.2)?


thresh = diabetes_data_drop.shape[0]*0.7
diabetes_data_drop = diabetes_data_drop.dropna(thresh=thresh, axis=1)
print(diabetes_data_drop.shape[1])


# Удалите из данных только те строки, в которых содержится более двух пропусков одновременно. 
# Чему равно результирующее число записей в таблице?

m = diabetes_data_drop.shape[1]
diabetes_data_drop = diabetes_data_drop.dropna(thresh=m-2, axis=0)
print(diabetes_data_drop.shape[0])

print(diabetes_data_drop.head())

# В оставшихся записях замените пропуски на медиану. Чему равно среднее значение в столбце SkinThickness? 
# Ответ округлите до десятых.

#создаем копию исходной таблицы
diabetes_data_drop_new = diabetes_data_drop.copy()
#создаем словарь имя столбца: число(признак) на который надо заменить пропуски
values = {
    'Pregnancies': diabetes_data_drop_new['Pregnancies'].median(),
    'Glucose': diabetes_data_drop_new['Glucose'].median(),
    'BloodPressure': diabetes_data_drop_new['BloodPressure'].median(),
    'BMI': diabetes_data_drop_new['BMI'].median(),
    'DiabetesPedigreeFunction': diabetes_data_drop_new['DiabetesPedigreeFunction'].median(),
    'Age': diabetes_data_drop_new['Age'].median(),
    'Outcome': diabetes_data_drop_new['Outcome'].median(),
}
#заполняем пропуски в соответствии с заявленным словарем
diabetes_data_drop_new = diabetes_data_drop_new.fillna(values)
#выводим результирующую долю пропусков
print(diabetes_data_drop_new.isnull().mean())
print(diabetes_data_drop_new['SkinThickness'].mean().round(1))

# Сколько выбросов найдёт классический метод межквартильного размаха в признаке SkinThickness?
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
outliers, _ = outliers_iqr_mod(diabetes_data_drop_new, 'SkinThickness')
print(outliers.shape[0])


def outliers_z_score_mod(data, feature, left=3, right=3, log_scale=False):
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned
outliers, _ = outliers_z_score_mod(diabetes_data_drop_new, 'SkinThickness')
print(outliers.shape[0])

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
outliers, _ = outliers_iqr_mod(diabetes_data_drop_new, 'DiabetesPedigreeFunction')
outliers_log, _ = outliers_iqr_mod(diabetes_data_drop_new, 'DiabetesPedigreeFunction', log_scale=True)
print(outliers.shape[0] - outliers_log.shape[0])