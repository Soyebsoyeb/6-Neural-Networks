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

