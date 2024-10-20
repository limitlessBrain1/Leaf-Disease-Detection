
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medicinal Plant Identification</title>
</head>
<body>
    <h1>Medicinal Plant Identification and Classification</h1>
    <p>This project focuses on the identification and classification of medicinal plants using image processing techniques and deep learning models. The system leverages advanced Convolutional Neural Networks (CNN) to accurately recognize and classify various medicinal plant species based on their leaf features.</p>

    <h2>Features</h2>
    <ul>
        <li><strong>High Accuracy</strong>: Achieved up to 99.66% accuracy in identifying medicinal plants using fine-tuned CNN models.</li>
        <li><strong>Multiple Models</strong>: Implemented and compared different deep learning architectures such as VGG16, MobileNet, and DenseNet for optimal performance.</li>
        <li><strong>Data Augmentation</strong>: Utilized data augmentation techniques to enhance the training dataset and improve model generalization.</li>
        <li><strong>Feature Fusion</strong>: Combined features from multiple CNN models to achieve better prediction accuracy.</li>
    </ul>

    <h2>Technologies Used</h2>
    <ul>
        <li>Programming Language: Python</li>
        <li>Deep Learning Framework: TensorFlow, Keras</li>
        <li>Models Implemented: VGG16, MobileNet, DenseNet, VGG19</li>
        <li>Image Processing: OpenCV, PIL</li>
        <li>Development Tools: Jupyter Notebook, Google Colab</li>
    </ul>

    <h2>Installation</h2>
    <p>To run this project locally, follow these steps:</p>
    <ol>
        <li>Clone the repository:</li>
        <pre><code>git clone https://github.com/limitlessBrain1/Leaf-Disease-Detection.git<br>
cd Leaf-Disease-Detection</code></pre>
        <li>Install the required dependencies:</li>
        <pre><code>pip install -r requirements.txt</code></pre>
        <li>Download or prepare your dataset of medicinal plant images.</li>
        <li>Run the training script:</li>
        <pre><code>python train.py</code></pre>
        <li>To evaluate the model, use:</li>
        <pre><code>python evaluate.py</code></pre>
    </ol>

    <h2>Usage</h2>
    <ul>
        <li><strong>Training</strong>: The <code>train.py</code> script trains the CNN models using the provided dataset. It also includes options for data augmentation and hyperparameter tuning.</li>
        <li><strong>Evaluation</strong>: The <code>evaluate.py</code> script evaluates the trained models and provides accuracy metrics.</li>
        <li><strong>Prediction</strong>: Use the <code>predict.py</code> script to classify new plant images. Simply pass the image file path as an argument.</li>
    </ul>

    <h2>Project Structure</h2>
    <pre><code>
├── data/
│   ├── train/            # Training data
│   └── test/             # Testing data
├── models/
│   ├── vgg16_model.h5    # Saved VGG16 model
│   ├── mobilenet_model.h5 # Saved MobileNet model
│   └── densenet_model.h5  # Saved DenseNet model
├── notebooks/
│   ├── Data_Preprocessing.ipynb
│   └── Model_Training.ipynb
├── scripts/
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── predict.py        # Prediction script
├── requirements.txt      # Python dependencies
└── README.md             # Project README
    </code></pre>

    <h2>Results</h2>
    <p>The models were tested on a dataset of medicinal plant images and demonstrated excellent performance, with the DenseNet model achieving the highest accuracy. Detailed results and analysis can be found in the <code>notebooks/Model_Training.ipynb</code> file.</p>

    <h2>Future Work</h2>
    <ul>
        <li>Model Optimization: Further optimization of the CNN models to reduce computational complexity.</li>
        <li>Real-Time Application: Implementation of a mobile application for real-time medicinal plant identification.</li>
        <li>Expanded Dataset: Increase the dataset to include more diverse plant species.</li>
    </ul>

    <h2>Contributing</h2>
    <p>Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or improvements.</p>

    <h2>License</h2>
    <p>This project is licensed under the MIT License.</p>
</body>
</html>

