import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def create_features(data):
    """
    Create polynomial features based on the given data.

    Args:
        data (pandas.DataFrame): The input data containing the features.

    Returns:
        pandas.DataFrame: The data with additional polynomial features.

    """
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    data_poly = pd.DataFrame(poly.fit_transform(data[['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']]),
                             columns=poly.get_feature_names(['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']))
    data = pd.concat([data, data_poly], axis=1)
    return data


def prepare_X(df, base, year_col='year', doors_col='number_of_doors', make_col='make',
              fuel_type_col='engine_fuel_type', transmission_col='transmission_type'):
    """
    Prepare the feature matrix X for a given dataframe.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - base (list): The list of base features.
    - year_col (str): The column name for the year.
    - doors_col (str): The column name for the number of doors.
    - make_col (str): The column name for the make.
    - fuel_type_col (str): The column name for the engine fuel type.
    - transmission_col (str): The column name for the transmission type.

    Returns:
    - X (numpy.ndarray): The feature matrix.
    """

    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df[year_col]
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df[doors_col] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df[make_col] == v).astype(int)
        features.append(feature)

    for v in ['regular_unleaded', 'premium_unleaded_(required)', 
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df[fuel_type_col] == v).astype(int)
        features.append(feature)

    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df[transmission_col] == v).astype(int)
        features.append(feature)

    df_num = df[features]
    df_num = df_num.fillna(0)
    return df_num
# X = df_num.values
#    return X