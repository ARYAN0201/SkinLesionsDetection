# Skin Lesions Detection

Skin Lesions Detection is a deep learning-based project aimed at identifying and classifying various types of skin lesions from medical images. Built using PyTorch, the model leverages convolutional neural networks (CNNs) and transfer learning techniques to achieve high accuracy in detecting potentially malignant skin conditions, such as melanoma. The system processes input images through a pipeline of preprocessing, augmentation, and classification, enabling reliable and efficient diagnosis support. Designed with flexibility in mind, the project can be easily extended for real-time applications using tools like Streamlit or Flask. This project serves both as a diagnostic aid and as a learning platform for exploring computer vision techniques in the medical imaging domain.


## Installation:

1. Clone the Repository:
   
   ``` bash
   git clone https://github.com/ARYAN0201/SkinLesionsDetection.git
   cd SkinLesionsDetection
   ```
   
2. Create a virtual environment:
   
   ``` bash
   conda create -n skinenv python=3.10
   conda activate skinenv
   ```
   
3. Install Dependencies:
   
   ``` bash
   pip install -r requirements.txt
   ```

## Usage:

To train the model and benchmark the results:

```bash
python benchmark.py
```

To predict for an image:

```bash
python predicting_image.py --model model_name --image path/to/image.jpg --checkpoint path/to/checkpoint.pth
```
