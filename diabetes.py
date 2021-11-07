import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd
df=pd.read_csv('diabetes.csv')
print(df.columns)
#df_req=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']] or
df_req=df.drop('Outcome',axis=1)
print(df_req.corr())
plt.figure(figsize=(10,8))
sb.heatmap(df.corr())
plt.show()
sb.pairplot(df,hue='Outcome')
plt.show()
y=df['Outcome']
x_train=df_req[:600]
x_test=df_req[600:]
y_train=y[:600]
y_test=y[600:]
model= LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
cm=confusion_matrix(y_test,pred)
print(cm)
'''[[TP FP]
   [FN TN]]'''
   #accuracy
print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[0][1]))
k=list(map(float,input('Enter values:').split()))
s=[]
s.append(k)
print(model.predict(s))