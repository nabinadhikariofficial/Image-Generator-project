This repository implements a Denoising Diffusion Probabilistic Model (DDPM) for image generation using TensorFlow and Keras. The model is trained on the Oxford-IIIT Pet dataset and includes a custom-implemented time-aware U-Net model with ResNet blocks and self-attention mechanism.


## How to Run

1.  **Clone the repository and dataset:**
    ```bash
    git clone https://github.com/Shreehar01/Image-Generator.git
    cd Image-Generator
    git clone https://github.com/ml4py/dataset-iiit-pet.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start training:**
    ```bash
    python main.py
    ```
    The script will create a directory named `epoch_outputs` and save generated images there every 10 epochs.