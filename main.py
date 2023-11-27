import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the Brooklyn homes dataset
brooklyn_homes = pd.read_csv("brooklyn_sales_map.csv")
neighborhood_mapping = pd.read_csv("neighborhood_mapping.csv")
inflation_data = pd.read_csv("Inflation.csv")
percapita_data = pd.read_csv("Percapita.csv")

# Exclude transactions with sale_price of $0 or nominal amounts
brooklyn_homes = brooklyn_homes[brooklyn_homes.sale_price > 1000]

# Group the data by year and calculate the total sales for each year
sales_by_year = brooklyn_homes.groupby(brooklyn_homes['sale_date'].str[:4])['sale_price'].sum()

# Plot the data
plt.plot(sales_by_year.index, sales_by_year.values)
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.title('Comparison of Sales Over the Years')
plt.show()

# Extract year and month from the sale_date column
brooklyn_homes['sale_date'] = pd.to_datetime(brooklyn_homes['sale_date'])
brooklyn_homes['Year'] = brooklyn_homes['sale_date'].dt.year
brooklyn_homes['Month'] = brooklyn_homes['sale_date'].dt.month

# Group the data by year and month, and calculate the total sales for each month
sales_by_month = brooklyn_homes.groupby(['Year', 'Month'])['sale_price'].sum()

# Plot the data with different colored lines for each year
fig, ax = plt.subplots(figsize=(10, 6))

for year in brooklyn_homes['Year'].unique():
    sales_data_year = sales_by_month.loc[year]
    ax.plot(sales_data_year.index.get_level_values('Month'), sales_data_year.values, label=f'Year {year}')

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_xlabel('Month')
ax.set_ylabel('Total Sales')
ax.set_title('Monthly Sales Comparison Over the Years')
ax.legend()
plt.show()

sales_by_neighborhood = brooklyn_homes.groupby('neighborhood')['sale_price'].sum()

# Sort neighborhoods by total sales in descending order
sales_by_neighborhood = sales_by_neighborhood.sort_values(ascending=False)

# Plot sales per neighborhood
plt.figure(figsize=(12, 8))
sales_by_neighborhood.plot(kind='bar', color='skyblue')
plt.xlabel('Neighborhood')
plt.ylabel('Total Sales')
plt.title('Total Sales per Neighborhood in Brooklyn')
plt.xticks(rotation=90)
plt.show()

# Group the data by year and calculate the average sale_price for each year
avg_price_by_year = brooklyn_homes.groupby('Year')['sale_price'].mean()

# Plot the comparison of sale_price variation over the years
plt.figure(figsize=(10, 6))
plt.plot(avg_price_by_year.index, avg_price_by_year.values, marker='o', color='orange', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Year')
plt.ylabel('Average Sale Price')
plt.title('Average Sale Price Variation Over the Years')
plt.grid(True)
plt.show()


# Sort the data by sale_price in descending order and select top 100 highest prices
top_100_highest_prices = brooklyn_homes.nlargest(100, 'sale_price')

# Group the top 100 highest prices by neighborhood and calculate the average price for each neighborhood
avg_price_top_100_by_neighborhood = top_100_highest_prices.groupby('neighborhood')['sale_price'].mean()

# Select the top 10 neighborhoods with the highest average prices from the top 100 highest prices
top_10_neighborhoods = avg_price_top_100_by_neighborhood.nlargest(10)

# Plot the top 10 neighborhoods and their average prices
plt.figure(figsize=(12, 6))
top_10_neighborhoods.plot(kind='bar', color='skyblue')
plt.xlabel('Neighborhood')
plt.ylabel('Average Sale Price')
plt.title('Top 10 Neighborhoods with Highest Average Sale Prices (Top 100 Highest Sale Price Houses)')
plt.xticks(rotation=45)
plt.show()

# Group the data by year and find the highest and lowest sale prices for each year
highest_prices = brooklyn_homes.groupby('Year')['sale_price'].max()
lowest_prices = brooklyn_homes.groupby('Year')['sale_price'].min()

# Plot the highest and lowest sale prices over the years
plt.figure(figsize=(10, 6))
plt.plot(highest_prices.index, highest_prices.values, marker='o', color='orange', label='Highest Price')
plt.plot(lowest_prices.index, lowest_prices.values, marker='o', color='green', label='Lowest Price (> $1000)')
plt.xlabel('Year')
plt.ylabel('Sale Price')
plt.title('Highest and Lowest Sale Prices Over the Years')
plt.legend()
plt.grid(True)
plt.show()

# Group the data by year and calculate the total number of homes sold per year
homes_sold_per_year = brooklyn_homes.groupby('Year').size().reset_index(name='Homes Sold')

# Ensure that 'Year' column is in both DataFrames and has the same name
inflation_data['Year'] = inflation_data['Year'].astype(int)

# Merge inflation data with homes sold data on 'Year' column
merged_data = pd.merge(homes_sold_per_year, inflation_data, on='Year', how='inner')

# Plot the inflation line and the total number of homes sold per year on the same graph
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting total number of homes sold per year (left y-axis)
ax1.plot(merged_data['Year'], merged_data['Homes Sold'], color='orange', marker='o', label='Homes Sold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Homes Sold', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')
ax1.set_title('Inflation and Homes Sold Over the Years')

# Create a second y-axis for inflation (right y-axis)
ax2 = ax1.twinx()
ax2.plot(merged_data['Year'], merged_data['Annual'], color='green', marker='o', label='Inflation')
ax2.set_ylabel('Inflation (Annual %)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Display legends for both lines
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Load and preprocess the time series data
# (Assuming the data is loaded into a DataFrame named 'time_series_data')

# Extract the date column from the dataset (assuming the date column is named 'sale_date')
dates = pd.to_datetime(brooklyn_homes['sale_date'])

# Generate synthetic time series data based on historical data
num_data_points = len(dates)
mean_sale_price = brooklyn_homes['sale_price'].mean()
std_dev_sale_price = brooklyn_homes['sale_price'].std()

# Generate random values following a normal distribution with mean and standard deviation from the actual data
synthetic_sale_prices = np.random.normal(loc=mean_sale_price, scale=std_dev_sale_price, size=num_data_points)

# Create a DataFrame with dates as index and synthetic sale prices
synthetic_time_series_data = pd.DataFrame(data={'sale_price': synthetic_sale_prices}, index=dates)

# Print the generated synthetic time series data
print(synthetic_time_series_data.head())


# Assuming 'synthetic_time_series_data' DataFrame contains synthetic sale prices with dates as the index

# Feature engineering: Extracting year and month from the index
synthetic_time_series_data['Year'] = synthetic_time_series_data.index.year
synthetic_time_series_data['Month'] = synthetic_time_series_data.index.month

# Creating a binary column 'Interest_Growth' to represent areas with growing interest (1) and decreasing interest (0)
synthetic_time_series_data['Interest_Growth'] = 0  # Default is decreasing interest

# Linear regression for each year to identify trends
for year in synthetic_time_series_data['Year'].unique():
    year_data = synthetic_time_series_data[synthetic_time_series_data['Year'] == year]

    X = np.arange(len(year_data)).reshape(-1, 1)
    y = year_data['sale_price'].values

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    predictions = model.predict(X)

    # Assessing trend by comparing the first and last sale prices in the year
    if predictions[-1] > predictions[0]:
        synthetic_time_series_data.loc[synthetic_time_series_data['Year'] == year, 'Interest_Growth'] = 1

# Visualizing the results
plt.figure(figsize=(10, 6))
plt.plot(synthetic_time_series_data.index, synthetic_time_series_data['sale_price'], label='Synthetic Sale Prices', color='blue')
plt.scatter(synthetic_time_series_data[synthetic_time_series_data['Interest_Growth'] == 1].index,
            synthetic_time_series_data[synthetic_time_series_data['Interest_Growth'] == 1]['sale_price'],
            label='Interest Growing', color='green', marker='o')
plt.scatter(synthetic_time_series_data[synthetic_time_series_data['Interest_Growth'] == 0].index,
            synthetic_time_series_data[synthetic_time_series_data['Interest_Growth'] == 0]['sale_price'],
            label='Interest Decreasing', color='red', marker='o')

plt.xlabel('Date')
plt.ylabel('Sale Price')
plt.title('Synthetic Time Series Data and Interest Assessment')
plt.legend()
plt.show()

print("Prediction Code")
housing_data = pd.read_csv('brooklyn_sales_map.csv')

# scatterplot visualisation
plt.scatter(x=housing_data['year_of_sale'], y=housing_data['sale_price'])
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.draw()

# highest selling property
housing_data.sort_values('sale_price').tail(1)
housing_data['sale_price'].describe().apply(lambda x: format(x, 'f'))
housing_data = housing_data[housing_data.sale_price > 0]

# more visualisations
bins = [-100000000, 20000, 40000, 60000, 80000, 100000, 1000000, 10000000, 500000000]
choices = ['$0-$200k', '$200k-$400k', '$400k-$600k', '$600k-$800k', '$800k-$1mlln', '$1mlln-$10mlln',
           '$10mlln-$100mlln', '$100mlln-$500mlln']
housing_data['price_range'] = pd.cut(housing_data['sale_price'], bins=bins, labels=choices)


def conv(year):
    return housing_data[housing_data['year_of_sale'] == year].groupby('price_range').size()


perc_total = [x / sum(x) * 100 for x in
              [conv(2003), conv(2004), conv(2005), conv(2006), conv(2007), conv(2008), conv(2009), conv(2010),
               conv(2011), conv(2012), conv(2013), conv(2014), conv(2015), conv(2016), conv(2017)]]
year_names = list(range(2003, 2018))
housing_df = pd.DataFrame(perc_total, index=year_names)
ax_two = housing_df.plot(kind='barh', stacked=True, width=0.80)
horiz_offset = 1
vert_offset = 1
ax_two.set_xlabel('Percentages')
ax_two.set_ylabel('Years')
ax_two.legend(bbox_to_anchor=(horiz_offset, vert_offset))
housing_data.groupby(['neighborhood', 'price_range']).size().unstack().plot.bar(stacked=True)
horiz_offset = 1
vert_offset = 1
plt.rcParams["figure.figsize"] = [40, 20]


# removing outliers
def remove_outlier(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    out_df = df.loc[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return out_df


housing_data = remove_outlier(housing_data, "sale_price")

# cleaning up columns with too many NAs
threshold = len(housing_data) * .75
housing_data.dropna(thresh=threshold, axis=1, inplace=True)

# more clean up
housing_data = housing_data.drop(
    ['APPBBL', 'BoroCode', 'Borough', 'BBL', 'price_range', 'PLUTOMapID', 'YearBuilt', 'CondoNo', 'BuiltFAR',
     'FireComp', 'MAPPLUTO_F', 'Sanborn', 'SanitBoro', 'Unnamed: 0', 'Version', 'block', 'borough', 'Address',
     'OwnerName', 'zip_code'], axis=1)

# if basement data is missing it might be safer to assume that whether or not the apartment/building is unknown which is represented by the number 5
housing_data['BsmtCode'] = housing_data['BsmtCode'].fillna(5)
# Community Area- not applicable or available if Na
housing_data[['ComArea', 'CommFAR', 'FacilFAR', 'FactryArea', 'RetailArea', 'ProxCode', 'YearAlter1', 'YearAlter2']] = \
housing_data[
    ['ComArea', 'CommFAR', 'FacilFAR', 'FactryArea', 'RetailArea', 'ProxCode', 'YearAlter1', 'YearAlter2']].fillna(0)
housing_data[
    ['XCoord', 'YCoord', 'ZipCode', 'LotType', 'SanitDistr', 'HealthArea', 'HealthCent', 'PolicePrct', 'SchoolDist',
     'tax_class_at_sale', 'CD', 'Council']] = housing_data[
    ['XCoord', 'YCoord', 'ZipCode', 'LotType', 'SanitDistr', 'HealthArea', 'HealthCent', 'PolicePrct', 'SchoolDist',
     'tax_class_at_sale', 'CD', 'Council']].apply(lambda x: x.fillna(x.mode()[0]))
# soft impute
from sklearn.impute import SimpleImputer
numeric_cols = housing_data.select_dtypes(include=[np.number]).columns
categorical_cols = housing_data.select_dtypes(exclude=[np.number]).columns

imputer_numeric = SimpleImputer(strategy='median')
imputer_categorical = SimpleImputer(strategy='most_frequent')

housing_data[numeric_cols] = imputer_numeric.fit_transform(housing_data[numeric_cols])
housing_data[categorical_cols] = imputer_categorical.fit_transform(housing_data[categorical_cols])


# change strings to ints to preprocess for ML algo
def strnums(cols):
    return dict(zip(set(housing_data[cols]), list(range(0, len(set(housing_data[cols]))))))


for columns in set(housing_data.select_dtypes(exclude='number')):
    housing_data[columns] = housing_data[columns].map(strnums(columns))
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

features = list(housing_data.drop(['sale_price'], axis=1))
y = housing_data.sale_price
X = housing_data[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
dummy_median = DummyRegressor(strategy='mean')
dummy_regressor = dummy_median.fit(X_train, y_train)
dummy_predicts = dummy_regressor.predict(X_test)
print("Model Accuracy:", dummy_regressor.score(X_test, y_test) * 100)
print('$', mean_absolute_error(y_test, dummy_predicts))

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have X_train, X_test, y_train, y_test, and features defined
# Replace this with your actual data

# Initialize Gaussian Naive Bayes model
nb_model = GaussianNB()

# Fit the model to the training data
nb_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_model.predict(X_test)

# Calculate R2 score
r2_nb = r2_score(y_test, y_pred)
print('R2 Score:', r2_nb)

# Feature Importance is not directly available in Gaussian Naive Bayes
# You may not have a direct feature importance measure like in tree-based models

# Visualize the R2 score
plt.bar(['Naive Bayes'], [r2_nb], align='center')
plt.title('R2 Score Comparison')
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.show()
