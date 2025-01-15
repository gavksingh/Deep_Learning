# Deep Learning Techniques: Exploratory and Comparative Analysis

This repository contains a series of deep learning experiments and analyses conducted as part of a graduate-level course in Machine Learning. The focus of this work is to explore, implement, and optimize various deep learning models on structured and image datasets.

## Repository Structure

| File Name                               | Description                                                                                 |
|-----------------------------------------|---------------------------------------------------------------------------------------------|
| **Data_Preprocessing_and_Exploratory_Analysis.ipynb** | Preprocessing the dataset, handling missing values, and performing exploratory data analysis (EDA). |
| **Neural_Network_Model_Training.ipynb** | Designing and training a neural network for classification tasks, including performance evaluation. |
| **Hyperparameter_Optimization.ipynb**   | Exploring the impact of different hyperparameters on model performance, such as dropout rate and optimizer choice. |
| **CNN_Model_Comparison.ipynb**          | Comparing a custom CNN with modified VGG-13 architecture for image classification tasks.     |
| **Bonus_ResNet_Model.ipynb**            | Implementation and evaluation of a ResNet-based model for advanced classification tasks.     |
| **Model_Interpretability.ipynb**        | Techniques for interpreting and understanding neural network predictions.                    |

## Key Highlights

### Data Preprocessing and EDA
- Cleaned the dataset by handling missing values and standardizing features.
- Performed visualization techniques like pair plots and correlation matrices to identify feature relationships.

### Neural Network Implementation
- Designed a multi-layer neural network with ReLU activation, dropout, and batch normalization for regularization.
- Achieved a test accuracy of 79.82% using optimized hyperparameters.

### Hyperparameter Tuning
- Experimented with dropout rates, batch sizes, and optimizers (Adam, RMSprop, SGD).
- Observed the best performance with a dropout rate of 0.5, batch size of 128, and RMSprop optimizer.

### CNN and VGG-13 Comparison
- Implemented a custom CNN and modified VGG-13 architecture.
- VGG-13 outperformed the CNN model with a test accuracy of 92% compared to 89%.

### Advanced Models and Techniques
- Implemented a ResNet-based model for enhanced feature extraction.
- Explored interpretability methods to analyze model predictions, including class-specific visualizations.

## Performance Metrics
- **Custom CNN**: Test accuracy = 89%, Precision = 87%, Recall = 88%
- **Modified VGG-13**: Test accuracy = 92%, Precision = 91.96%, Recall = 91.62%, F1-Score = 91.57%
- ROC curves and confusion matrices were generated to validate the models’ performance.

## Tools and Libraries
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Other Libraries**: NumPy, pandas, Matplotlib, Seaborn, scikit-learn

