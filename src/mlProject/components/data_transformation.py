import os 
import pandas as pd 
import numpy as np 

# Project Directory Related Imports 
from mlProject import logger
from mlProject.entitiy.config_entity import DataTransformationConfig
# ML Related Imports 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
import joblib

class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config=config 
        self.labelencoder=LabelEncoder()
        self.preprocessor = None 
    
        # Extract Schema 
        self.columns = self.config.schema["COLUMNS"]
        self.target_column = self.config.target_column
        self.feature_columns = [col for col in self.columns.keys() if col != self.target_column]

        # Identify column types from schema 
        self.numerical_columns = [col for col, dtype in self.columns.items() if dtype in ["int64","float64"] and col != self.target_column]
        self.categorical_columns = [col for col, dtype in self.columns.items() if dtype == "object" and col != self.target_column]
        
        logger.info(f"Initialized DataTransformation with schema")
        logger.info(f"Target column: {self.target_column}")
        logger.info(f"Numerical columns: {self.numerical_columns}")
        logger.info(f"Categorical columns: {self.categorical_columns}")
        
    def drop_id_column(self,data):  

        if "id" in self.columns:
            if "id" in data.columns:
                data = data.drop("id",axis=1)

            if "id" in self.numerical_columns:
                self.numerical_columns.remove("id")
                
            if "id" in self.feature_columns:
                self.feature_columns.remove("id")
                logger.info("ID column dropped")
        return data 
    
    def handle_missing_values(self,data):
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")

            for col in self.numerical_columns:
                data[col].fillna(data[col].median(),inplace=True)
                logger.info(f"Handle numerical missing values in {col} with median imputation")
            
            for col in self.categorical_columns:
                if col in data.columns and data[col].isnull().Any():
                    data[col].fillna(data[col].mode()[0],inplace=True)
                    logger.info(f"Handled categorical missing values in {col} with mode")
        else:
            logger.info("No missing values found")
        
        return data 
    
    def detect_and_handle_outliers(self,data):
        outlier_info = {}

        for col in self.numerical_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR 
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_info[col] = len(outliers)
                data[col] = data[col].clip(lower_bound,upper_bound)
        
        if outlier_info:
            logger.info(f"Handles outiners : {outlier_info}")
        else:
            logger.info(f"No significant outliers detected")

        return data 

    def encode_target_variable(self, data):
        if self.target_column in data.columns:
            if data[self.target_column].dtype == "object":
                data["Heart Disease"] = self.labelencoder.fit_transform(data["Heart Disease"])
                logger.info("Encoded target variable ")

        return data 

    def handle_categorical_features(self,data):

        for col in self.categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
            logger.info(f"Encoded Categorical Column : {col}")
        return data 

    def scale_features(self,X_train,X_test):
        numerical_cols = [col for col in self.numerical_columns if col in X_train.columns]

        if numerical_cols:

            # Create preprocessors 
            self.preprocessor = StandardScaler()

            # Fit on training data and transform on both 
            X_train_scaled = self.preprocessor.fit_transform(X_train[numerical_cols])
            X_test_scaled = self.preprocessor.transform(X_test[numerical_cols])

            # Convert back to dataframe with column names 
            X_train_scaled = pd.DataFrame(X_train_scaled,columns = numerical_cols,index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled,columns=numerical_cols,index=X_test.index)


            other_cols = [col for col in X_train.columns if col not in numerical_cols]
            for col in other_cols:
                X_train_scaled[col] = X_train[col].values
                X_test_scaled[col] = X_test[col].values 

            logger.info("Scaled numerical features using standard scaler")
            return X_train_scaled,X_test_scaled 
        return X_train,X_test 
    def save_preprocessor(self):

        preprocessor_path = os.path.join(self.config.root_dir,"preprocessor.pkl")
        joblib.dump(self.preprocessor,preprocessor_path)
        logger.info(f"Saved preprocessor to {preprocessor_path}")

        encoder_path = os.path.join(self.config.root_dir,"label_encoder.pkl")
        joblib.dump(self.labelencoder,encoder_path)
        logger.info(f"Label Encoder saved to {encoder_path}")

        # Save column info for inference 
        column_info = {
            "feature_columns": self.feature_columns,
            "numerical_columns":self.numerical_columns,
            "categorical_columns":self.categorical_columns,
            "target_columns":self.target_column 
        }

        column_info_path = os.path.join(self.config.root_dir,"column_info.pkl")
        joblib.dump(column_info,column_info_path)
        logger.info(f"Saved column information to {column_info_path}")

    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)
        logger.info(f"Loaded data with shape : {data.shape}")
        
        data = self.drop_id_column(data)

        data = self.handle_missing_values(data)

        data = self.detect_and_handle_outliers(data)

        data = self.handle_categorical_features(data) 

        X = data.drop(self.target_column,axis=1)
        y = data[self.target_column]

        X = X[self.feature_columns]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        X_train_scaled , X_test_scaled = self.scale_features(X_train,X_test)

        # Combine feature and target for saving 
        train_data  = X_train_scaled.copy()
        train_data[self.target_column] = y_train

        test_data = X_test_scaled.copy()
        test_data[self.target_column] = y_test 

        train_data.to_csv(os.path.join(self.config.root_dir,"train.csv"),index=False)
        test_data.to_csv(os.path.join(self.config.root_dir,"test.csv"),index=False)  

        self.save_preprocessor()
        logger.info("Splitting data into training and testing sets")

        logger.info(f"Training data : {train_data.shape}")
        logger.info(f"Testing data : {test_data.shape}")      

        print(f"Training data : {train_data.shape}")
        print(f"Testing data : {test_data.shape}") 
