import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class CustomerExitModel:
    def __init__(self):
    # Initialize models with hyperparameters to prevent overfitting
        self.rfmodel = RandomForestClassifier(
            random_state=45, 
            n_estimators=100,  # Limit the number of trees
            max_depth=10,      # Control tree depth
            min_samples_split=10, 
            min_samples_leaf=5
        )
        self.dtmodel = DecisionTreeClassifier(
            random_state=44,
            max_depth=10,      # Limit tree depth
            min_samples_split=10,
            min_samples_leaf=5
        )
        self.lrmodel = LogisticRegression(
            random_state=42,
            penalty='l2',      # Regularization
            C=1.0,             # Strength of regularization
            max_iter=200       # Ensure convergence
        )
        self.transformer = None

        self.feature_columns = [
        "creditscore", "age", "tenure", "balance", "numofproducts", "estimatedsalary",
        "geography_Germany", "geography_Spain", "gender_Male", "hascrcard", "isactivemember"
        ]

    def train(self):
        # Load and preprocess the dataset
        df = pd.read_csv("churn.csv", index_col=0)
        df.columns = map(str.lower, df.columns)

        # Encode categorical variables and drop unnecessary columns
        df = pd.get_dummies(df, columns=["geography", "gender"], drop_first=True)
        df = df.drop(["customerid", "surname"], axis=1)

        # Separate features and target
        cat_df = df[["geography_Germany", "geography_Spain", "gender_Male", "hascrcard", "isactivemember"]]
        y = df["exited"]
        X = df.drop(["exited", "geography_Germany", "geography_Spain", "gender_Male", "hascrcard", "isactivemember"], axis=1)

        # Scale numerical features
        self.transformer = RobustScaler().fit(X)
        X_scaled = self.transformer.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        X = pd.concat([X, cat_df], axis=1)

        # Handle class imbalance and split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        smk = SMOTETomek()
        X_train, y_train = smk.fit_resample(X_train, y_train)

        # Train the Random Forest model
        self.rfmodel.fit(X_train, y_train)
        self.dtmodel.fit(X_train, y_train)
        self.lrmodel.fit(X_train, y_train)

        
        for name, model in [("Random Forest", self.rfmodel), ("Decision Tree", self.dtmodel), ("Logistic Regression", self.lrmodel)]:
        # Make predictions for both training and test sets
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

        # Calculate accuracy for training and test sets
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)

        # Determine fit status
            if train_accuracy < 0.7 and test_accuracy < 0.7:
                fit_status = "Underfitting"
            elif train_accuracy - test_accuracy > 0.1:
                fit_status = "Overfitting"
            else:
                fit_status = "Normal Fit"

        # Print results
            print(f"{name}:")
            print(f"  Training Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")
            print(f"  Fit Status: {fit_status}\n")
            
        self.evaluate_model(X_test,y_test)


    def evaluate_model(self, X_test, y_test):
        # Predictions for all models
        y_pred_dt = self.dtmodel.predict(X_test)
        y_pred_rf = self.rfmodel.predict(X_test)
        y_pred_lr = self.lrmodel.predict(X_test)

        # Calculate accuracy for each model
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)

        # Calculate precision, recall, F1-score
        precision_rf = precision_score(y_test, y_pred_rf)
        precision_dt = precision_score(y_test, y_pred_dt)
        precision_lr = precision_score(y_test, y_pred_lr)

        recall_rf = recall_score(y_test, y_pred_rf)
        recall_dt = recall_score(y_test, y_pred_dt)
        recall_lr = recall_score(y_test, y_pred_lr)

        f1_rf = f1_score(y_test, y_pred_rf)
        f1_dt = f1_score(y_test, y_pred_dt)
        f1_lr = f1_score(y_test, y_pred_lr)

        # Print the evaluation metrics for each model
        print(f"Random Forest - Accuracy: {accuracy_rf:.2f}, Precision: {precision_rf:.2f}, Recall: {recall_rf:.2f}, F1: {f1_rf:.2f}")
        print(f"Decision Tree - Accuracy: {accuracy_dt:.2f}, Precision: {precision_dt:.2f}, Recall: {recall_dt:.2f}, F1: {f1_dt:.2f}")
        print(f"Logistic Regression - Accuracy: {accuracy_lr:.2f}, Precision: {precision_lr:.2f}, Recall: {recall_lr:.2f}, F1: {f1_lr:.2f}")



    def predict(self, input_data):
        # Ensure the model is trained
        predict_report = {"Random Forest" : 0,"Decision Tree" : 0,"Logistic Regression" : 0}

        if not self.transformer or not (self.rfmodel or self.dtmodel or self.lrmodel):
            raise ValueError("Model or transformer not initialized. Train the model first.")

            # Convert input data to DataFrame
        user_input = pd.DataFrame([input_data], columns=self.feature_columns)

            # Scale numeric features
        numeric_features = ["creditscore", "age", "tenure", "balance", "numofproducts", "estimatedsalary"]
        user_input[numeric_features] = self.transformer.transform(user_input[numeric_features])
        
        for name, model in [("Random Forest", self.rfmodel), ("Decision Tree", self.dtmodel), ("Logistic Regression", self.lrmodel)]:
            # Predict using the trained model
            prediction = model.predict(user_input)
            predict_report[name] = int(prediction[0])
        return predict_report
            
