(1)

Spiral Classification with Neural Networks
This project demonstrates how to classify spiral data using a neural network built with TensorFlow/Keras. The model learns to distinguish between three intertwined spirals, a classic non-linear classification problem






The project contains:

Spiral data generation

Data preprocessing and scaling

Neural network implementation

Model training with callbacks

Performance evaluation

Visualization of results

Dataset
Synthetic spiral data with 3 classes

999 total points (333 per class)

80-20 train-test split

Features: 2D coordinates (x, y)

Target: Class labels (0, 1, 2)







Training
Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Metrics: Accuracy

Callbacks:

Early Stopping (patience=20)

ReduceLROnPlateau (factor=0.1, patience=10)

Epochs: 200 (stops early if no improvement)






Results
Typical performance:

Training Accuracy: ~99%

Validation Accuracy: ~98%

Confusion Matrix shows excellent class separation

