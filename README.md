# Fashion-MNIST Data Analysis
## Description
This repository contains analysis and training a deep learning model on infamous Fashion-MNIST dataset 
which predicts 10 classses like t-shirt, torso, sneaker etc based on the trained model.

## Table of Contents
* Data Analysis
* Model Developement
  * Model Architecture
  * Trainning and Optimization
* Result
* Usage (Must Read)

## Data Analysis

Upon obtaining the dataset, a comprehensive analysis was conducted to gain insights into its composition. The dataset comprises 10 distinct classes, each containing approximately 7,000 data points. This balanced distribution ensures that the model's training is not skewed towards any particular class, reducing the risk of overfitting or underfitting.

The dataset contains 785 columns, with the first column serving as the class label ('label'). The remaining 784 columns correspond to pixel values, collectively representing 28x28 pixel images. These pixel values are crucial as they form the input data for the machine learning model.

![image](https://github.com/sasori-s/Fashion-MNIST/assets/89335680/0d8c7abd-9179-47f3-8afa-65174aae4e9f)


In this analysis, I sought to understand the distribution of these classes and the nature of the images. This knowledge provides valuable context for building and evaluating our mode

![image](https://github.com/sasori-s/Fashion-MNIST/assets/89335680/16b8f931-cc0e-4a0c-a1f6-ab9134cd58f1)


## Model Developement
For the task of classifying sustainable apparel products, I designed a machine learning model using TensorFlow and Keras. The model's architecture plays a crucial role in achieving accurate and reliable predictions.

### Model Architecture
I chose a Sequential model for its simplicity and effectiveness in handling structured data. The model consists of several layers designed to process the input data efficiently:

Input Layer: I started with a Flatten layer to transform the input data. In our dataset, each example has 784 features, representing the pixel values of a 28x28 pixel image. This layer effectively reshapes the data to be compatible with the model.

Hidden Layers: I introduced two dense hidden layers with 64 and 128 units, respectively, utilizing the Rectified Linear Unit (ReLU) activation function. These layers allow the model to learn complex patterns and representations from the data.

Output Layer: The final dense layer contains 10 units, each corresponding to one of the 10 classes in the dataset. I used the softmax activation function in this layer, which is suitable for multi-class classification tasks. It provides the probability distribution over the classes, helping us make accurate predictions.

### Trainning and Optimization
I compiled the model using the 'adam' optimizer and 'sparse_categorical_crossentropy' as the loss function, which is appropriate for multi-class classification. To monitor the model's performance during training, I tracked the 'accuracy' metric.

The model was trained on a training dataset, and I conducted training for 10 epochs. Each epoch represents one pass through the entire training dataset. This iterative process allows the model to learn and improve its ability to classify sustainable apparel products.

After training, I evaluated the model's performance on a separate test dataset. The evaluation provides crucial insights into the model's accuracy and its ability to generalize to unseen data.


## Result
The model predicts the images like this
![image](https://github.com/sasori-s/Fashion-MNIST/assets/89335680/5cc2d33d-a8ba-4317-b7aa-bc502ba9a086)



## Requirements
I have provided a requirement.txt file, First you need to create a virtual environment.
You can install all the packages listed in a requirements.txt file using the pip command. Here's how to do it:

Open a command prompt or terminal.

Navigate to the directory where your requirements.txt file is located using the cd command, then type this command `pip install -r requirements.txt
`
## Usage
In 'evaluate_model.py' please provide path for your model in `model_path` and test dataset in `test_path` variable, For example 
![image](https://github.com/sasori-s/Fashion-MNIST/assets/89335680/2b5bc66e-6324-43f8-b11e-d572a51ff55d)

To run the script run this command `!python evaluate_model.py --model_path '/content/evaluate_model.py' --test_file '/content/fashion_mnist_data/fashion-mnist_test.csv'
`. Please replace the `/content/evaluate_model.py` and `/content/fashion_mnist_data/fashion-mnist_test.csv` with your actual path for model and test data respectively. 
After running the model you should be able to download output.txt file. Which contains the accuracy score and necessary information about model summary. 
