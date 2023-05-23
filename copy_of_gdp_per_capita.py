# -*- coding: utf-8 -*-
import itertools as iter
import numpy as np
import itertools as iter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv')
df.describe()

# Drop the column "1960"
df = df.drop('1960', axis=1)

# Select the columns to be used for clustering
cols = ['1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

# Impute NaN values with the mean of the corresponding column
df[cols] = df[cols].fillna(df[cols].mean())

# Normalize the data
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])

# Use KMeans clustering algorithm to cluster the data
kmeans = KMeans(n_clusters=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[cols])

df['Cluster'].value_counts()

# new_df = df.drop(df[df['Cluster'] == 0].index)
# Define the list of cluster values to drop
clusters_to_drop = [0, 7, 4, 6, 3]

# Create a boolean mask that selects the rows with the specified cluster values
mask = df['Cluster'].isin(clusters_to_drop)

# Drop the rows that match the mask
new_df = df.drop(df[mask].index)

new_df['Cluster'].value_counts()

# Use logical slicing to select the data for plotting
data = new_df.loc[df['Cluster'].isin([5, 2, 1,9,8])]


# # Use logical slicing to select the data for plotting
# data = df.loc[df['Cluster'].isin([5, 2])]

sns.scatterplot(data=data, x='2019', y='2000', hue='Cluster')
plt.scatter(x=kmeans.cluster_centers_[:, 48], y=kmeans.cluster_centers_[:, 49], marker='x', s=200, linewidths=3, color='black')
plt.xlabel('2019')
plt.ylabel('2000')
plt.title('KMeans Clustering')
plt.show()

# Use logical slicing to select the data for plotting
data = new_df.loc[df['Cluster'].isin([1,9,8])]

sns.scatterplot(data=data, x='2019', y='2000', hue='Cluster')
plt.scatter(x=kmeans.cluster_centers_[:, 48], y=kmeans.cluster_centers_[:, 49], marker='x', s=200, linewidths=3, color='black')
plt.xlabel('2019')
plt.ylabel('2000')
plt.title('KMeans Clustering')
plt.show()

print(df.columns)

# Find one country from each cluster
countries = []
for i in [1, 2, 5, 8, 9]:
    cluster_data = df[df['Cluster'] == i]
    country = cluster_data.sample(1)['Country Code'].values[0]
    countries.append(country)

# Compare the countries from one cluster to find similarities and differences
cluster_data = df[df['Cluster'] == 0]  # replace 0 with the cluster number you want to compare
cluster_data = cluster_data.drop(['Country Code', 'Cluster'], axis=1)
mean_values = cluster_data.mean()
max_values = cluster_data.max()
min_values = cluster_data.min()

print("Countries selected from each cluster:", countries)
print("Mean values for the selected cluster:\n", mean_values)
print("Maximum values for the selected cluster:\n", max_values)
print("Minimum values for the selected cluster:\n", min_values)

# Define the data for the five selected countries
data = {
    'SSA': [-2.45, 2.72, -1.98, -3.87, -0.66, -0.56, -1.08, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1., -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -0.28, 0.37, 0.03, 0.23, -0.26],
    'GHA': [-1.54, 5.13, -1.98, -3.87, -0.66, -0.56, -1.08, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -0.28, 0.37, 0.03, 0.23, -0.26],
    'VGB': [-2.45, 2.72, -1.98, -3.87, -0.66, -0.56, -1.08, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -0.28, 0.37, 0.03, 0.23, -0.26],
    'KAZ': [-2.45, 2.72, -1.98, -3.87, -0.66, -0.56, -1.08, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -0.28, 0.37, 0.03, 0.23, -0.26],
    'BIH': [-2.45, 2.72, -1.98, -3.87, -0.66, -0.56, -1.08, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -1.16, -0.28, 0.37, 0.03, 0.23, -0.26]
}

# Define the x-axis values (years)
years = range(1961, 2022)

# Create a line chart for each country
for country, values in data.items():
    # Select a random sample of 10 values from the y-axis values
    sample = random.sample(values, 10)
    # Plot the sample values against the x-axis values
    plt.plot(years[:len(sample)], sample, label=country)

# Add a legend and axis labels
plt.legend()
plt.xlabel('Year')
plt.ylabel('GDP per capita growth (annual %)')

# Show the chart
plt.show()

cluster_8_data = df[df['Cluster'] == 8]
cluster_9_data = df[df['Cluster'] == 9]
cluster_6_data = df[df['Cluster'] == 5]
cluster_2_data = df[df['Cluster'] == 2]
cluster_1_data = df[df['Cluster'] == 1]
# Extract the country codes for the data points in cluster 8
country_codes_8 = cluster_8_data['Country Code'].tolist()
country_codes_9 = cluster_9_data['Country Code'].tolist()
country_codes_5 = cluster_5_data['Country Code'].tolist()
country_codes_2 = cluster_2_data['Country Code'].tolist()
country_codes_1 = cluster_1_data['Country Code'].tolist()
# Print the list of country codes
print("Country codes in cluster 8:", country_codes_8)
print("Country codes in cluster 9:", country_codes_9)
print("Country codes in cluster 5:", country_codes_5)
print("Country codes in cluster 2:", country_codes_2)
print("Country codes in cluster 1:", country_codes_1)

# Assuming you have a DataFrame called "new_df" with a column called "Cluster"
# Select data from cluster 5 only
cluster_8_data = new_df.loc[new_df['Cluster'] == 8]

# Use the describe method to get summary statistics for the cluster 5 data
# cluster_8_summary = cluster_5_data.describe()
cluster_8_data.describe()

# Load the dataset
df = pd.read_csv('API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv')

# Drop the column "1960"
df = df.drop('1960', axis=1)

# Select the columns to be used for modeling
cols = ['1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

# Impute NaN values with the mean of the corresponding column
df[cols] = df[cols].fillna(df[cols].mean())

# Define a simple model function
def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit the model to the data
x_data = np.array(df['2019'])
y_data = np.array(df['1991'])
popt, pcov = curve_fit(model_func, x_data, y_data)

# Make predictions for future years
x_pred = np.arange(2022, 2031)
y_pred = model_func(x_pred, *popt)

# Plot the data and the model predictions
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_pred, y_pred, label='Model')
plt.xlabel('2019')
plt.ylabel('1991')
plt.title('Simple Model Fitting')
plt.legend()
plt.show()

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    """
    # Calculate the function values for all combinations of +/- sigma
    values = []
    for s in [-sigma, sigma]:
        for p in np.meshgrid(*[[0, s]] * len(param)):
            values.append(func(*p))
    
    # Determine the minimum and maximum values
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Return the upper and lower limits
    return min_val, max_val


# Load the dataset
df = pd.read_csv('API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv')

# Drop the column "1960"
df = df.drop('1960', axis=1)

# Select the columns to be used for clustering
cols = ['1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

# Impute NaN values with the mean of the corresponding column
df[cols] = df[cols].fillna(df[cols].mean())

# Normalize the data
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])

# Use KMeans clustering algorithm to cluster the data
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[cols])

# Use logical slicing to select the data for plotting
data = df.loc[df['Cluster'].isin([1, 2, 3])]

# Plot the data
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=data, x='2019', y='2020', hue='Cluster')
plt.scatter(x=kmeans.cluster_centers_[:, 48], y=kmeans.cluster_centers_[:, 49], marker='x', s=200, linewidths=3, color='black')
plt.xlabel('2019')
plt.ylabel('2020')
plt.title('KMeans Clustering')

# Calculate the error ranges for the cluster centers
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i} center: {center}")
    param = ['x', 'y']
    sigma = 1
    lower, upper = err_ranges(center, func=lambda x, y: x + y, param=param, sigma=sigma)
    print(f"Error range for Cluster {i} center: [{lower}, {upper}]")

plt.show()



