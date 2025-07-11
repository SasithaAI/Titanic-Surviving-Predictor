import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from data_preprocessing import load_and_preprocess_data

def train_survival_model(df):
    """Trains a Logistic Regression model on the Titanic dataset."""
    X = df.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)

    # Save the trained model and the list of features
    with open('survival_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model_features.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    print("Model training complete and saved as survival_model.pkl")
    print("Model features saved as model_features.pkl")

if __name__ == '__main__':
    url1 = "https://www.openml.org/data/get_csv/9900/titanic.arff"
    url2 = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    processed_df = load_and_preprocess_data(url1, url2)
    if processed_df is not None:
        train_survival_model(processed_df)
