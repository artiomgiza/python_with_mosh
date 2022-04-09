import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model1 = DecisionTreeClassifier()
model1.fit(X, y)

# store model:
joblib.dump(model1, 'music-remommender.joblib')

# load a stored model
model2 = joblib.load('music-remommender.joblib')

predictions = model2.predict([[21, 1]])
predictions
