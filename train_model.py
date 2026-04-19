import pandas as pd
df=pd.read_csv('driver_attention_data.csv')
print(df.head())
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X=df[['face_present','face_size','face_centered']]
y=df['label']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
accuracy=model.score(x_test,y_test)
print("Model Accuracy:",accuracy)
prediction=model.predict([[1,500,0]])
print("Predicted label:",prediction[0])
