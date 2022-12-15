#Подключаем библиотеки
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd

#подключаем датасет
df = pd.read_csv('diamonds_moded.csv', sep=';')

print(df.info())
print(df.head())

# Проверим наличие пропусков в данных
print( df.isna().sum() )

# Пропусков мало - заполним их модой
df['color'].fillna(df['color'].mode()[0], inplace=True)

print( df.isna().sum() )

print(df.info())

print(df['cut'].value_counts())

# def cut(x):
#     if x == 'Ideal':
#         return 1
#     elif x == 'Premium':
#         return 2
#     elif x == 'Very Good':
#         return 3
#     elif x == 'Good':
#         return 4
#     elif x == 'Fair':
#         return 5
#     return x

# df['cut'] = df['cut'].apply(cut)

print(df.info())

print(df.head())

# Находим категориальные колонки 
print(df.info())

#и используем LabelEncoder для перевода в численные значения
encoder = preprocessing.LabelEncoder()
df['cut'] = encoder.fit_transform(df['cut'])
df['color'] = encoder.fit_transform(df['color'])
df['clarity'] = encoder.fit_transform(df['clarity'])

#убеждаемся, что все данные в числовом формате
print(df.info())
print(df.head())


# cat_columns = [cname for cname in df.columns if df[cname].dtype == "object"]
# encoder = preprocessing.LabelEncoder()
# for col in cat_columns:
#     df[col] = encoder.fit_transform(df[col])

print(df.head())

#выделям основные данные и целевой признак
X = df.drop('price', axis=1)
y = df['price']

#разделяем данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#тренируем модель
lr = LinearRegression()
lr.fit(X_train, y_train)

#делаем предсказания
y_pred = lr.predict(X_test)
print(y_pred)

#узнаём качество предсказаний
print(lr.score(X_test, y_test))

df2 = pd.DataFrame()

df2['y_test']=y_test
df2['y_pred']=y_pred

print(df2[['y_test', 'y_pred']])

import matplotlib.pyplot as plt

plt.plot(1,1)
plt.show()