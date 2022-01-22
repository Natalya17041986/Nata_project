import pandas as pd

student_data = pd.read_csv('data/students_performance.csv', sep=',')

# данные о скольки студентах содержатся в таблице
print(student_data.shape[0])

# какой балл по письму у студента под номером 155
print(student_data.loc[155, 'writing score'])

print(student_data.info())

# средний балл по математике
print(int(student_data['math score'].mean()))

# какая рассовая группа самая крупная
print(student_data['race/ethnicity'].mode())

# какой средний балл по чтению у студентов которые посещали курсы подготовки к экзаменам
print(round(student_data[student_data['test preparation course'] == 'completed']['reading score'].mean()))

# сколько студентов получило по математике 0
print(student_data[student_data['math score'] == 0].shape[0])

print(student_data[student_data['lunch'] == 'standard']['math score'].mean())
print(student_data[student_data['lunch'] == 'free/reduced']['math score'].mean())

# какой процент студентов у которых родители имеют высшее образование
print(student_data["parental level of education"].value_counts(normalize=True))

a = student_data[student_data['race/ethnicity'] == 'group A']['writing score'].median()
b = student_data[student_data['race/ethnicity'] == 'group C']['writing score'].mean()
print(round(abs(a - b)))