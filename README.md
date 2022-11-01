# Neural-Networks-and-Deep-Learning-Models

## Description
Exploring and implementing neural networks using the TensorFlow platform in Python. We explore the major costs and benefits of different neural networks and compare these costs to traditional machine learning classification and regression models. Additionally, we practice implementing neural networks and deep neural networks across a number of different datasets, including image, natural language, and numerical datasets. Finally, we learn how to store and retrieve trained models for more robust uses.

### Challenge Overview
* Use a machine learning model and neural network model to build and predict the success of a venture paid by Alphabet Soup Company.
* Create a binary classifier to project which applicants are likely to be successful if they received future funding from Alphabet Soup. 

### Approach
* Used a CSV file with more than 34,000 organizations that have received various amounts of funding from Alphabet Soup over the years.
* First, import, analyze, clean, and preprocess a “real-world” classification dataset. 
* Then select, design, and train a binary classification model of your choosing. 
* Finally, optimize model training and input data to achieve desired model performance.

### Results

#### Data Preprocessing for a Neural Network

The following preprocessing steps have been performed:

* The EIN and NAME columns have been dropped
* The columns with more than 10 unique values have been grouped together
* The categorical variables have been encoded using one-hot encoding
* The preprocessed data is split into features and target arrays
* The preprocessed data is split into training and testing datasets
* The numerical values have been standardized using the StandardScaler() module


##### 1) What variable(s) are considered the target(s) for your model?

"IS_SUCCESSFUL" was the target varialbe (y) for the model

##### 2) What variable(s) are considered the feature(s) for your model?

The following are all part of the features maxtrix (X)
1. APPLICATION_TYPE<br>
2. AFFILIATION<br>
3. CLASSIFICATION<br>
4. USE_CASE<br>
5. ORGANIZATION<br>
6. STATUS<br> 
7. INCOME_AMT<br> 
8. SPECIAL_CONSIDERATIONS<br> 
9. ASK_AMT<br>

##### 3) What variable(s) are neither targets nor features, and should be removed from the input data?

EIN and NAME were not part of the inital model.

#### Compile, Train, and Evaluate the Model

TensorFlow was used to design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset.

The neural network model developed using Tensorflow Keras contains working code that performs the following steps:

* The number of layers, the number of neurons per layer, and activation function are defined
* An output layer with an activation function is created
* There is an output for the structure of the model
* There is an output of the model’s loss and accuracy
* The model's weights are saved every 5 epochs to a checkpoints folder
* The results are saved to an HDF5 file in the Trained_Models folder

#### Optimize the Model

##### 1) How many neurons and layers did you select for your neural network model? Why?

The initial model included the following: 
1. An input layer<br>
2. First hidden layer (Number of neurons was three times the number of inputs - following a general rule of thumb). Also the inital model used "relu" activation - as a default choice<br>
3. Second hidden layer (Number of neurons was half the number of inputs - interest in having fewer neurons any additional layers). Also the inital model used "relu" activation - as a default choice<br>
4. Output layer with a "sigmoid" activation function as this was a binary classifier

##### 2) Were you able to achieve the target model performance? What steps did you take to try and increase model performance?

Yes, the optimized model did achieve the target model performance even after optimizing model. The accurate rate was 77%.

The optimized model included an additional hidden layer (sandwiched between intial model's hidden layer 1 & 2). This additional hidden layer had the same number of neurons as the number of inputs. The optimized model also used "tahn" activation based on the values in X_train_scaled rangeing from -1 to 1.

In order to exceed the benchmark of 75% model accuracy the model was optimized by exploring the following methods:

* Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model

* Dropping more or fewer columns.

* Creating more bins for rare occurrences in columns.

* Increasing or decreasing the number of values for each bin.

* Adding more neurons to a hidden layer

* Adding more hidden layers.

* Using different activation functions for the hidden layers.

* Adding or reducing the number of epochs to the training regimen.

##### 3) If you were to implement a different model to solve this classification problem, which would you choose? Why?

Since AlphabetSoup Charity's dataset is tabular, a random forest classifier is the recommended model based on performance, speed, explainability and simplicity of setup.

**Compare to Other Models Random Forest Vs. Deep Learning Model**<br><br>
Random forest classifiers are a type of ensemble learning model that combines multiple smaller models into a more robust and accurate model. Random forest models use a number of weak learner algorithms (decision trees) and combine their output to make a final classification decision. Structurally speaking, random forest models are very similar to their neural network counterparts. Random forest models have been a staple in machine learning algorithms for many years due to their robustness and scalability. Both output and feature selection of random forest models are easy to interpret, and they can easily handle outliers and nonlinear data.

Random forest algorithms are beneficial because they:

* Are robust against overfitting as all of those weak learners are trained on different pieces of the data.
* Can be used to rank the importance of input variables in a natural way.
* Can handle thousands of input variables without variable deletion.
* Are robust to outliers and nonlinear data.
* Run efficiently on large datasets.

### Things I Learned
* Compare the differences between the traditional machine learning classification and regression models and the neural network models.
* Describe the perceptron model and its components.
* Implement neural network models using TensorFlow.
* Explain how different neural network structures change algorithm performance.
* Preprocess and construct datasets for neural network models.
* Compare the differences between neural network models and deep neural networks.
* Implement deep neural network models using TensorFlow.
* Save trained TensorFlow models for later use.

### Software/Tools
Jupyter Notebook, Python, TensorFlow
