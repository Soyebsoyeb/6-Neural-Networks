🧠 Spiral Classification with Neural Networks
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

✅ Validation: ~98%

🧩 Confusion Matrix: Shows clear class separation with excellent performance!

