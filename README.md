(1) 🧠 Spiral Classification with Neural Networks
This project shows how to classify spiral-shaped data using a neural network built with TensorFlow/Keras.
It tackles a classic non-linear classification problem involving three intertwined spirals! 🌀🌀🌀

📦 What's Inside?
✅ Spiral data generation
✅ Data preprocessing & scaling 🔄
✅ Neural network implementation 🤖
✅ Model training with smart callbacks ⚙️
✅ Performance evaluation 📊
✅ Cool visualizations 🎨

📊 Dataset Details
📌 Type: Synthetic spiral data

🎯 Classes: 3 (labels: 0, 1, 2)

🔢 Points: 999 (333 per class)

✂️ Split: 80% Training / 20% Testing

📍 Features: 2D coordinates (x, y)

🏋️ Model Training
⚙️ Optimizer: Adam

🧮 Loss Function: Sparse Categorical Crossentropy

📈 Metric: Accuracy

🧠 Callbacks:
⏹️ EarlyStopping: patience=20

📉 ReduceLROnPlateau: factor=0.1, patience=10

⏳ Max Epochs: 200 (training stops early if no improvement)

🏆 Results
📌 Typical Accuracy

✅ Training: ~99%




(2) # Neural Network from Scratch (Forward Pass)

A minimal implementation of a 2-layer neural network for multi-class classification, demonstrating the forward pass, activation functions, and loss calculation.

## Features
- **Layers**: `Dense` (fully connected), `ReLU`, `Softmax`.
- **Loss**: Categorical cross-entropy.
- **Data**: Spiral dataset (3 classes, 100 samples).

## Code Structure
1. **Data Generation**: `spiral_data()` creates non-linear, separable data.
2. **Layers**:
   - `Layer_Dense`: Linear transformation (`WX + b`).
   - `Activation_ReLU`: Non-linearity via `max(0, x)`.
   - `Activation_Softmax`: Outputs class probabilities.
3. **Loss**: `Loss_CategoricalCrossentropy` computes cross-entropy.
4. **Metrics**: Accuracy via `np.argmax` comparison.

# Create network
dense1 = Layer_Dense(2, 3)      # Input: 2D, Output: 3 neurons
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)      # Output: 3 classes
activation2 = Activation_Softmax()

# Forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Evaluate
loss = loss_function.calculate(activation2.output, y)
accuracy = np.mean(np.argmax(activation2.output, axis=1) == y)

✅ Validation: ~98%

🧩 Confusion Matrix: Shows clear class separation with excellent performance!




# SPIRAL DATASET RESOLVED BY BACKPROPAGATION (WITHOUT OPTIMIZER)

This is a complete implementation of a 2-layer neural network with forward and backward propagation, designed specifically for classifying spiral data 🌪️. It demonstrates core deep learning concepts using only NumPy, without any high-level frameworks.

⚙️ Exact Code Features
🧱 Layers
Layer_Dense (Fully-connected layer):

✅ Weight initialization: 0.01 * np.random.randn(...)

✅ Bias initialization: np.zeros(...)

🔄 Forward & backward propagation supported

⚡ Activations
Activation_ReLU:

Forward pass ✔️

Backward pass with gradient clipping 🔁

Activation_Softmax:

Stable implementation (max subtraction for numerical stability) 🧮

Combined: Softmax + CrossEntropy for optimized gradients 🚀

📉 Loss
Loss_CategoricalCrossentropy:

Accepts raw class labels: [0, 1, 2] 🔢

Accepts one-hot encoded labels: [[1,0,0], ...] ✅

Numerical stability with output clipping: [1e-7, 1 - 1e-7] 🛡️

📊 Data
Uses: nnfs.datasets.spiral_data(samples=100, classes=3) 🌪️

Visualized via Matplotlib:
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')

The script will
✅ Generate spiral data
✅ Run forward propagation
✅ Compute loss and accuracy
✅ Perform backpropagation

🖨️ Expected Output:

First 5 outputs:
[[0.33 0.33 0.33]
 [0.33 0.33 0.33]
 ...]

loss: 1.0986
acc: 0.34
🔍 Key Technical Details

🔁 Forward Pass
dense1.forward(X)                     # Input → Hidden (2→3)
activation1.forward(dense1.output)   # ReLU Activation
dense2.forward(activation1.output)   # Hidden → Output (3→3)
loss = loss_activation.forward(dense2.output, y)  # Softmax + Loss

🔁 Backward Pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
📐 Gradient Calculations
Weights: inputs.T @ dvalues

Biases: np.sum(dvalues, axis=0)

Inputs: dvalues @ weights.T

📈 Visualization
Spiral data example:
🌀 (Code will generate and plot automatically with Matplotlib)

