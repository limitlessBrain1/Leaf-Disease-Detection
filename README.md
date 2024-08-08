This project focuses on the identification and classification of medicinal plants using image processing techniques and deep learning models.
The system leverages advanced Convolutional Neural Networks (CNN) to accurately recognize and classify various medicinal plant species based on their leaf features.

Features
High Accuracy: Achieved up to 99.66% accuracy in identifying medicinal plants using fine-tuned CNN models.
Multiple Models: Implemented and compared different deep learning architectures such as VGG16, MobileNet, and DenseNet for optimal performance.
Data Augmentation: Utilized data augmentation techniques to enhance the training dataset and improve model generalization.

Feature Fusion: Combined features from multiple CNN models to achieve better prediction accuracy.

Technologies Used
Programming Language: Python
Deep Learning Framework: TensorFlow, Keras
Models Implemented: VGG16, MobileNet, DenseNet, VGG19
Image Processing: OpenCV, PIL
Development Tools: Jupyter Notebook, Google Colab
Installation
To run this project locally, follow these steps:

Clone the repository:

bash

git clone https://github.com/limitlessBrain1/Leaf-Disease-Detection.git
cd Leaf-Disease-Detection
Install the required dependencies:

bash

pip install -r requirements.txt
Download or prepare your dataset of medicinal plant images.

Run the training script:

bash

python train.py
To evaluate the model, use:

bash

python evaluate.py
Usage
Training: The train.py script trains the CNN models using the provided dataset. It also includes options for data augmentation and hyperparameter tuning.
Evaluation: The evaluate.py script evaluates the trained models and provides accuracy metrics.
Prediction: Use the predict.py script to classify new plant images. Simply pass the image file path as an argument.

Project Structure

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

Results
The models were tested on a dataset of medicinal plant images and demonstrated excellent performance, with the DenseNet model achieving the highest accuracy. 
Detailed results and analysis can be found in the notebooks/Model_Training.ipynb file.

Future Work
Model Optimization: Further optimization of the CNN models to reduce computational complexity.
Real-Time Application: Implementation of a mobile application for real-time medicinal plant identification.
Expanded Dataset: Increase the dataset to include more diverse plant species.
Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or improvements.

License
