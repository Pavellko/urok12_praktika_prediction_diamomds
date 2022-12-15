
# Визуализация
from matplotlib import pyplot as plt

plt.scatter(x_train, y_train,color='b')
plt.plot(x_test, predictions,color='black', linewidth=3)
plt.xlabel('Рост отцов в дюймах')  
plt.ylabel('Рост сыновей в дюймах') 
plt.show()

