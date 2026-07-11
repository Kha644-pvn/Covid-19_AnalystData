import pandas as pd

url = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
df = pd.read_csv(url)

df.to_csv("covid_data.csv", index=False)
