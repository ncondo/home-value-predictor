from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, boxcox_normmax
from scipy.special import boxcox1p
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from home_value_predictor.datasets.dataset import Dataset


RAW_DATA_DIRNAME = Dataset.data_dirname()/"raw"/"home_data"
RAW_DATA_FILENAME = RAW_DATA_DIRNAME/"train.csv"

PROCESSED_DATA_DIRNAME = Dataset.data_dirname()/"processed"/"home_data"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "processed_train.parquet"


class HomeDataset(Dataset):

    def __init__(self):
        if not os.path.exists(RAW_DATA_FILENAME):
            print('File not found')


    def load_data(self, processed=True):
        if processed:
            if not os.path.exists(PROCESSED_DATA_FILENAME):
                self.process_data()
            df = pd.read_parquet(PROCESSED_DATA_FILENAME)
        else:
            df = pd.read_csv(RAW_DATA_FILENAME)

        return df

    
    def process_data(self):
        """Process the data and store as ready for training"""
        df = pd.read_csv(RAW_DATA_FILENAME)

        target = df['SalePrice']
        df = df.drop('SalePrice', axis=1)

        # apply log transformation to the target variable
        target = np.log1p(target)

        # drop redundant columns
        df.drop(['Id'], axis=1, inplace=True)

        # convert data types
        numeric_features = list(df.select_dtypes(
                            include=[np.number]).columns.values)
        categ_features = list(df.select_dtypes(
                            include=['object']).columns.values)

        for col in numeric_features:
            df[col] = df[col].astype(float)

        # replace NaNs in categorical features with "None"
        df[categ_features] = df[categ_features].apply(
                            lambda x: x.fillna("None"), axis=0)

        # impute four numerical features with zero
        for col in ('LotFrontage','GarageYrBlt','GarageArea','GarageCars'):
            df[col].fillna(0.0, inplace=True)

        # impute other numerical features with median
        df[numeric_features] = df[numeric_features].apply(
                                lambda x: x.fillna(x.median()), axis=0)

        df = self.get_features(df)
        df['SalePrice'] = target

        df.to_parquet(PROCESSED_DATA_FILENAME)


    def get_features(self, df):
        """Do feature engineering"""

        # create combination of features
        df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + 
                                    df['1stFlrSF'] + df['2ndFlrSF'])

        df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + 
                                df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

        df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 
                                df['EnclosedPorch'] + df['ScreenPorch'] + 
                                df['WoodDeckSF'])

        # create boolean features
        df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

        # transform numerical features that should be considered as strings
        df['MSSubClass'] = df['MSSubClass'].apply(str)
        df['YrSold'] = df['YrSold'].astype(str)
        df['MoSold'] = df['MoSold'].astype(str)
        df['YrBltAndRemod'] = df['YrBltAndRemod'].astype(str)

        # Transform numerical columns with skewness factor > 0.5
        numeric_features = list(df.select_dtypes(include=[np.number]).columns.values)
        skew_features = df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
        high_skew = skew_features[skew_features > 0.5]
        skew_index = high_skew.index
        for i in skew_index:
            df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))

        # label encoding
        df = pd.get_dummies(df)

        # feature scaling
        scaler = RobustScaler()
        for col in numeric_features:
            df[[col]] = scaler.fit_transform(df[[col]])

        return df


    def split_data(self, df, test_size=0.2, random_state=42):
        target = df['SalePrice']
        df = df.drop('SalePrice', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(df,
                                                            target,
                                                            test_size=test_size,
                                                            random_state=random_state)

        return X_train, X_test, y_train, y_test




    
