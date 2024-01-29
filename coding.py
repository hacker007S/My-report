

import pandas as pd
import numpy as np

def read_file(file_path):
    """ Function to read a file and return a DataFrame. """
    return pd.read_csv(file_path)

def load_and_process_data(file_path):
    """ Load and process the dataset. """
    # Load the data
    df = read_file(file_path)

    # Display the first few rows and info of the dataframe
    print(df.head())
    print(df.info())

    # Use .describe() to get statistical summary of the dataframe
    print(df.describe())

    # Melt the DataFrame
    melted_df = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Value')

    # Convert the 'Year' column to numeric and sort the DataFrame
    melted_df['Year'] = pd.to_numeric(melted_df['Year'], errors='coerce')
    melted_df = melted_df.sort_values(by=['Country Name', 'Year'])

    # Transpose the DataFrame
    transposed_df = melted_df.transpose()

    return transposed_df

def skewness(dist):
    """ Calculates the centralised and normalised skewness of dist. """
    average = np.mean(dist)
    std_dev = np.std(dist)
    return np.sum(((dist - average) / std_dev)**3) / len(dist)

def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """
    average = np.mean(dist)
    std_dev = np.std(dist)
    return np.sum(((dist - average) / std_dev)**4) / len(dist) - 3.0

def bootstrap(dist, function, confidence_level=0.90, nboot=10000):
    """ Bootstrap to get the uncertainty of a statistical function applied to dist. """
    fvalues = np.array([])

    for _ in range(nboot):
        random_sample = np.random.choice(dist, len(dist), replace=True)
        fvalues = np.append(fvalues, function(random_sample))

    qlow = 0.5 - confidence_level / 2.0
    qhigh = 0.5 + confidence_level / 2.0
    return np.quantile(fvalues, qlow), np.quantile(fvalues, qhigh)

if __name__ == '__main__':
    file_path = 'worlbank.csv'  # Replace with actual file path
    transposed_df = load_and_process_data(file_path)
    print(transposed_df)
