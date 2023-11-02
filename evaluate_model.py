import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

model_path = '/content/Fashion-MNIST_sq.h5'
test_path = '/content/fashion_mnist_data/fashion-mnist_test.csv'
model = keras.models.load_model(model_path)

def load_and_preprocess_test_data(test_folder):
    df = pd.read_csv(test_folder)
    y = df.iloc[:,0]
    X = df.iloc[:, 1: ]

    return X, y

def evaluate_model(model_path, test_path):
    correct_predictions = 0
    X, y = load_and_preprocess_test_data(test_path)
    total_examples = len(X)
    y_true = []
    y_pred = []

    # Iterate through all test examples
    for i in range(total_examples):
        test_example = X.iloc[i].to_numpy()  # Get the feature values for the i-th test example
        test_example = test_example.reshape(1, 784)  # Reshape for prediction

        # Make a prediction
        pred = model.predict(test_example)
        
        # Get the predicted class index
        predicted_class_index = np.argmax(pred[0])
        
        # Get the actual class label (ground truth)
        actual_class_label = y.iloc[i]
        y_pred.append(predicted_class_index)
        y_true.append(actual_class_label)

      

    # Calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the file containing test data")
    args = parser.parse_args()

    model_path = args.model_path
    test_folder = args.test_file

    try:
        accuracy = evaluate_model(model_path, test_folder)
        # model_summary = model.summary()
        # model_summary = str(model_summary)
        accuracy = str(accuracy)

        # Create and write to the output.txt file
        with open('output.txt', 'w') as output_file:
            output_file.write("Model Architecture:\n")
            model_summary = model.summary(print_fn=lambda x: output_file.write(x + '\n'))
            output_file.write(f"Test Accuracy: {accuracy}\n")
            output_file.write("\nAdditional Insights and observations:\n")
            output_file.write("\n1. The dataset's remarkable class balance with approximately 7,000 samples\nper class minimizes the risk of underfitting or overfitting,\nensuring a robust foundation for model training.\n")
            output_file.write("\n2. Reshaping the 784-pixel images was a crucial preprocessing step,\nenhancing data compatibility and model training efficiency.\n")
            output_file.write("\n3. Recognizing visually similar classes like pullovers, coats, and shirts\ncan be challenging. In these cases, leveraging human expertise in a human-in-the-loop approach\ncan improve model accuracy.\n")
            output_file.write("\n4. Future work can focus on refining the model, integrating more features,\nand further engaging human expertise to advance sustainable apparel classification.\n")
        print("Model evaluation completed. Results written to output.txt.")
    except FileNotFoundError:
        print("Error: The provided folder doesn't exist or is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
    sys.exit(0)



