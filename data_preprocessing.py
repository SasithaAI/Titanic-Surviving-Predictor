import pandas as pd
import requests

def load_and_preprocess_data(url1, url2):
    """Downloads, loads, cleans, and preprocesses the Titanic dataset."""
    try:
        response = requests.get(url1)
        if response.status_code == 200:
            with open("titanic.csv", "wb") as f:
                f.write(response.content)
            print("Dataset downloaded from primary source and saved as titanic.csv")
        else:
            print(f"Primary download failed. Status code: {response.status_code}")
            response = requests.get(url2)
            if response.status_code == 200:
                with open("titanic.csv", "wb") as f:
                    f.write(response.content)
                print("Dataset downloaded from alternative source and saved as titanic.csv")
            else:
                print(f"Alternative download failed. Status code: {response.status_code}")
                raise Exception("Both download attempts failed.")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

    titanic_df = pd.read_csv("titanic.csv")

    # Handle null values
    median_age = titanic_df['Age'].median()
    titanic_df['Age'] = titanic_df['Age'].fillna(median_age)

    mode_embarked = titanic_df['Embarked'].mode()[0]
    titanic_df['Embarked'] = titanic_df['Embarked'].fillna(mode_embarked)

    titanic_df = titanic_df.drop('Cabin', axis=1)

    # Convert qualitative features
    categorical_cols = ['Sex', 'Embarked']
    titanic_df = pd.get_dummies(titanic_df, columns=categorical_cols, drop_first=True)

    return titanic_df

if __name__ == '__main__':
    url1 = "https://www.openml.org/data/get_csv/9900/titanic.arff"
    url2 = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    processed_df = load_and_preprocess_data(url1, url2)
    if processed_df is not None:
        print("\nProcessed DataFrame head:")
        print(processed_df.head())
        print("\nProcessed DataFrame info:")
        processed_df.info()
