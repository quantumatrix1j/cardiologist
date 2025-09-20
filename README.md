# Cardiac MRI Prediction Model

This project uses a Convolutional Neural Network (CNN) to predict whether a cardiac MRI scan shows a "Normal" or "Sick" heart.

## Setup Instructions

1.  **Clone the Repository**
    ```bash
    git clone [your-repository-url]
    cd [your-project-folder]
    ```

2.  **Download the Dataset**
    - Download the dataset from Kaggle: [https://www.kaggle.com/datasets/danialsharifrazi/cad-cardiac-mri-dataset](https://www.kaggle.com/datasets/danialsharifrazi/cad-cardiac-mri-dataset)
    - Unzip it and make sure the `cad-cardiac-mri-dataset` folder is in the main project directory.

3.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Required Libraries**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run a Prediction

1.  Open the `predict.py` file.
2.  Find the `test_image_path` variable and replace the example path with the full path to any image file from the dataset on your computer.
3.  Run the script from your terminal:
    ```bash
    python predict.py
    ```
The model will then load and print its prediction for your chosen image.