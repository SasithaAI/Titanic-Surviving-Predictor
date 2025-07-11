import pandas as pd
import pickle

def predict_survival_probability(input_data):
    """
    Predicts the survival probability based on user input.

    Args:
        input_data: A dictionary containing user input for each feature.

    Returns:
        The predicted probability of survival (class 1), or None if model/features not found.
    """
    try:
        # Load the trained model and model features
        with open('survival_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_features.pkl', 'rb') as f:
            model_features = pickle.load(f)

    except FileNotFoundError:
        print("Error: Model or features file not found. Please train the model first.")
        return None

    # Create a DataFrame from the user input, ensuring correct column order
    input_df = pd.DataFrame([input_data], columns=model_features)

    # Predict the survival probability
    survival_probability = model.predict_proba(input_df)[:, 1]

    return survival_probability[0]

if __name__ == '__main__':
    # Example usage (replace with actual user input)
    example_input = {
        'Pclass': 1,
        'Age': 30.0,
        'SibSp': 1,
        'Parch': 0,
        'Fare': 50.0,
        'Sex_male': 1,
        'Embarked_Q': 0,
        'Embarked_S': 1
    }
    probability = predict_survival_probability(example_input)
    if probability is not None:
        print(f"Predicted survival probability: {probability:.4f}")

    # Example with different input
    example_input_2 = {
        'Pclass': 3,
        'Age': 25.0,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 10.0,
        'Sex_male': 0, # Female
        'Embarked_Q': 0,
        'Embarked_S': 1
    }
    probability_2 = predict_survival_probability(example_input_2)
    if probability_2 is not None:
        print(f"Predicted survival probability (example 2): {probability_2:.4f}")
