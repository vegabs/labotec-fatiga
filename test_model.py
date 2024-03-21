import joblib
import numpy as np

# Load the exported SVM model
svm_model = joblib.load('svm_modelv2.pkl')

# Define a function to make predictions using the loaded model
def predict_category(input_vector):
    # Convert input_vector to a numpy array if it's not already
    input_vector = np.array(input_vector).reshape(1, -1)  # Reshape to 2D array
    
    # Make predictions using the loaded model
    predicted_category = svm_model.predict(input_vector)
    
    return predicted_category[0]

# Example usage
input_vector = [0.338724103073748,0.33126691777222306,0.30300484511731696,0.28878518750018267,0.2891222885183676,0.2976183657533733,0.31650505915708826,0.31191857036743587,0.3226028856896457,0.3214211230708298,0.32433139646806963,0.3392410531442782,0.3437079365526936,0.32912240287556804,0.3119013699926035]  # Example input vector with 15 elements
predicted_category = predict_category(input_vector)
print(f"Predicted Category: {predicted_category}")