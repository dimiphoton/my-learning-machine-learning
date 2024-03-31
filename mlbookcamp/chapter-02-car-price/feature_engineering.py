import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def create_features(data):
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    data_poly = pd.DataFrame(poly.fit_transform(data[['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']]),
                             columns=poly.get_feature_names(['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']))
    data = pd.concat([data, data_poly], axis=1)
    return data