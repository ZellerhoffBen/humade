import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

path = "../melb_data.csv"

melbourne_home_data = pd.read_csv(path)

df = melbourne_home_data.head(1000).dropna(axis=0)

features = df.select_dtypes(include=["int64", "float64"]).columns

print(features)

X = df[features].drop(columns=["Price"])
y = df.Price

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = RandomForestRegressor(random_state=1)

model.fit(train_X, train_y)

predictions = model.predict(val_X)

mae = mean_absolute_error(predictions, val_y)

print(mae)
