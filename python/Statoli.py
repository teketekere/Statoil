import pandas as pd


train = pd.read_json("./data/train.json/data/processed/train.json")
print(train.head())
print(train.isnull().sum())
