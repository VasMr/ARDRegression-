from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ARDRegression
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt


diabetes = datasets.load_diabetes()
target = diabetes.target
df_target = pd.DataFrame(target, columns=['progressios'])
data = diabetes.data
feature_names = diabetes.feature_names
df_data = pd.DataFrame(data, columns=feature_names)

x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.1) #random_state=

model = ARDRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)

r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)

print('R2 :', r2)
print('RMSE :', rmse)
print('MAE :', mae)

x = np.arange(y_test.shape[0])
plt.title('Comparison of measured and predicted values')
plt.ylabel('progression')
plt.plot(x, y_test, label='y_test')
plt.plot(x, pred, label='pred')
plt.legend()
plt.show()
