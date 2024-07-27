import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Rajesh\Downloads\dataset1.csv")
data.head()

data.corr()
plt.matshow(data.corr())
plt.colorbar()


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)


from sklearn.linear_model import LogisticRegression
linear_model = LogisticRegression()
linear_model.fit(x_train,y_train)

y_pred=linear_model.predict(x_test)


plt.scatter(y_test,y_pred)
plt.xlabel("test values")
plt.xlabel("prediction values")


from sklearn.svm import SVC
# Create and Train the Support Vector Machine (Regression) using radial basis function
svr_rbf = SVC()
svr_rbf.fit(x_train, y_train)
svr_rbf.fit(x_train,y_train)

y_predsvm=svr_rbf.predict(x_test)


from sklearn import metrics
import numpy as np

print('logistic:', metrics.accuracy_score(y_test, y_pred))
print('logistic:', metrics.accuracy_score(y_test, y_predsvm))



print(svr_rbf.predict([[138,	8.6,	560,	7.46,	0.62,	0.7,	5.9,	0.24,	0.31,	0.77,	8.71,	0.11]]))
