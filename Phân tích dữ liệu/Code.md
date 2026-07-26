```python
countries = [
    'China', 'United Kingdom', 'Vietnam',
    'India', 'United States', 'Brazil', 'South Africa'
]
# 1. sort theo thời gian
df = df.sort_values(['country', 'date'])

#------- Phân tích mối tương quan giữa new_cases và new_deaths--------
# làm mượt dữ liệu sau khi tạo cột mức độ nguy hiểm
df['death_rate_smooth'] = (
    df.groupby('country')['death_rate']
    .rolling(7)
    .mean()
    .reset_index(0, drop=True)
)

df_filtered = df[df['country'].isin(countries)]
g = sns.FacetGrid(
    df_filtered,
    col='country',
    col_wrap=3,
    height=4,
    sharey=False
)
g.map_dataframe(
    sns.lineplot,
    x='date',
    y='death_rate_smooth'
)
g.set_titles("{col_name}")
g.set_axis_labels("Date", "Death Rate")
plt.show()

#---Phân tích mối tương quan giữa tỉ lệ tủ vong và tỉ lệ tiêm vacxin--
df['Ti_Le_Tu_Vong_smooth'] = (
    df.groupby('country')['Ti_Le_Tu_Vong']
    .rolling(7)
    .mean()
    .reset_index(0, drop=True)
)

df['Ti_Le_Tiem_smooth'] = (
    df.groupby('country')['Ti_Le_Tiem']
    .rolling(7)
    .mean()
    .reset_index(0, drop=True)
)

# vẽ biểu đồ
for country in countries:
    data = df[df['country'] == country]

    plt.figure(figsize=(10,5))
    plt.plot(data['date'], data['Ti_Le_Tu_Vong_smooth'], label='Death Rate')
    plt.plot(data['date'], data['Ti_Le_Tiem_smooth'], label='Vaccination Rate')

    plt.title(country)
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

#-Phân tích số người được tiêm đủ vắc xin so với số người được tiêm đủ vắc xin------------
df['Ti_Le_Tiem_Du_VacXin_smooth'] = (
    df.groupby('country')['Ti_Le_Tiem_Du_VacXin']
    .rolling(7)
    .mean()
    .reset_index(0, drop=True)
)

for country in countries:
    data = df[df['country'] == country]

    plt.figure(figsize=(10,5))

    plt.plot(data['date'], data['Ti_Le_Tiem_Du_VacXin_smooth'],
             label='Fully Vaccinated Ratio')

    plt.title(country)
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.legend()
    plt.xticks(rotation=45)

    plt.show()


# xuất ra file csv
# df_new.to_csv('data_filtered.csv',index=False)

```
