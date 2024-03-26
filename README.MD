# StackGAN

This repository contains the implementation of StackGAN, short for Stacked Generative Adversarial Network, is an advanced approach in the field of computer vision and machine learning, particularly in the domain of image generation.

## Getting Started

Follow these steps to use this implementation:

### Prerequisites

Ensure you have Python installed on your system. This code is compatible with Python 3.9 and newer versions.

### Dataset

For training and testing the StackGAN, you'll need a dataset. I used some Pokemon dataset which you download using the following link:

[Pokemon Dataset](https://www.kaggle.com/datasets/thedevastator/pokemon-llava-images-and-text-descriptions)

After downloading, place the dataset in an appropriate directory within the your project structure, such as "./data/images".

Next step, is to download processed embeddings text descriptions, i used CLIP model for this purpose. The link for embeddings is below, and you should place embeddings.pkl file in the "./data/embedding/"

### Installation

1. **Clone the repository** to your local computer:

    ```
    git clone https://github.com/dykyivladk1/StackGAN.git
    ```


2. **Install the required dependencies**. It's recommended to create and use a virtual environment:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Training model**

    To train a model for custom dataset, navigate to "codes" and use the following command:
    
    ```
    python scripts/train.py --stage <1> --yaml_path <yaml path> --data_path <data_path> --embedding_path <embedding_path>
    ```

4. **Evaluate sample** 

    To evaluate the model and generate some samples, navigate to "codes" directory and run the following command:

    ```
    python sample.py --yaml_path <yaml path> --prompt <prompt>
    ```

5. **Thanks for your attention.**
