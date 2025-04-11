from sklearn.decomposition import PCA
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split

class CustomerExitModel:
    def __init__(self):
        
        # Initialize models with hyperparameters
        self.rfmodel = RandomForestClassifier(
            random_state=45, 
            n_estimators=100,
            max_depth=10,
            min_samples_split=10, 
            min_samples_leaf=5
        )
        self.dtmodel = DecisionTreeClassifier(
            random_state=44,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        self.sgdmodel = SGDClassifier(
            random_state=42,
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3
        )

        self.transformer = None
        self.pca = None
        self.feature_columns = [
            "creditscore", "age", "tenure", "balance", "numofproducts", "estimatedsalary", "gender_Male", "hascrcard", "isactivemember"
        ]
        self.trained_models = {}  # Dictionary to store trained models

    def train(self):
        # Load and preprocess the dataset
        df = pd.read_csv("churn.csv", index_col=0)
        df.columns = map(str.lower, df.columns)

        # Encode categorical variables and drop unnecessary columns
        df = pd.get_dummies(df, columns=["gender"], drop_first=True)
        df = df.drop(["customerid", "surname","geography"], axis=1)

        # Separate features and target
        cat_df = df[[ "gender_Male", "hascrcard", "isactivemember"]]
        y = df["exited"]
        X = df.drop(["exited", "gender_Male", "hascrcard", "isactivemember"], axis=1)

        #print(X.head(),"\n")
        # Scale numerical features
        self.transformer = RobustScaler().fit(X)
        X_scaled = self.transformer.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        X = pd.concat([X, cat_df], axis=1)
        print(X.head())
        # Apply PCA
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        X_pca = self.pca.fit_transform(X)

        # Handle class imbalance and split data
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.20, random_state=42)

        # Train models and store them in `self.trained_models`
        for name, model in [("Random Forest", self.rfmodel), ("Decision Tree", self.dtmodel), ("SGD Classifier", self.sgdmodel)]:
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            y_pred_test = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred_test)


            print(f"{name}:")
            print(f"  Test Accuracy: {test_accuracy:.2f}")
            print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

    def predict(self, input_data):
        # Validate if models and preprocessing are initialized
        if not self.transformer or not self.pca or not self.trained_models:
            raise ValueError("Model, PCA, or transformer not initialized. Train the model first.")

        # Convert input data to DataFrame
        user_input = pd.DataFrame([input_data], columns=self.feature_columns)

        # Scale numerical features
        numeric_features = ["creditscore", "age", "tenure", "balance", "numofproducts", "estimatedsalary"]
        user_input[numeric_features] = self.transformer.transform(user_input[numeric_features])
        print(user_input.head())
        # Apply PCA transformation
        user_input_pca = self.pca.transform(user_input)

        # Predict using each model
        predictions = {}
        for name, model in self.trained_models.items():
            predictions[name] = int(model.predict(user_input_pca)[0])

        return predictions
