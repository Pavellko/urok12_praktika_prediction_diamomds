# Подключение библиотек
import pandas as pd
from sklearn.linear_model import LinearRegression

# Подключение датасета
df = pd.read_csv('data.csv' ,delim_whitespace=True)

# Загрузка основных данных и целевого признака
x_train = df.drop('Son', axis=1)
y_train = df['Son']

# Тренировка модели
lm = LinearRegression()
lm.fit(x_train, y_train)

# Тестовые данные
x_test = [[72.8],[61.1],[67.4],[70.2],
			[75.6],[60.2],[65.3],[59.2]]

# Предсказание
predictions = lm.predict(x_test)
print (predictions)

from matplotlib import pyplot as plt

plt.scatter(x_train, y_train,color='b')
plt.plot(x_test, predictions,color='black', linewidth=3)
plt.xlabel('Рост отцов в дюймах')  
plt.ylabel('Рост сыновей в дюймах') 
plt.show()