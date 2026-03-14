import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Marks": [20,25,30,40,50,60,65,75]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

hours = pd.DataFrame([[5]], columns=["Hours"])
pred = model.predict(hours)

print("Predicted Marks:", pred[0])
