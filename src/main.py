import pandas as pd
import preprocess as preP

# Config to show all columns
pd.set_option('display.max_columns', None)

# load data
data = preP.load_data("../data/diabetes.csv")

print(data.sample(5))