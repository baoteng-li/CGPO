<<<<<<< HEAD

<h1 align="center"> Curriculum Group Policy Optimization:<br>Adaptive Sampling for Unleashing the
Potential of Text-to-Image Generation </h1>

The paper will be announced later.

## 🚀 Quick Started
### 1. Environment Set Up
Clone this repository and install packages.
```bash
git clone https://github.com/baoteng-li/CGPO.git
cd CGPO
conda create -n cgpo python=3.10.16
pip install -e .
```
### 2. Reward Preparation
We adopted the same reward model processing approach as Flow-GRPO. Since each reward model may rely on different versions, combining them in one Conda environment can cause version conflicts. To avoid this, we adopt a remote server setup inspired by ddpo-pytorch. You only need to install the specific reward model you plan to use. For more information, please refer to [Flow-GRPO](https://github.com/yifan123/flow_grpo).

#### GenEval
Please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [reward-server](https://github.com/yifan123/reward-server).

#### OCR
Please install paddle-ocr:
```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```
Then, pre-download the model using the Python command line:
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
```

#### Pickscore
PickScore requires no additional installation.

### 3. Start Training
Single-node training:
```bash
bash scripts/single_node/cgpo.sh
```

## ✨ Important Hyperparameters
You can adjust the parameters in `config/grpo.py` to tune different hyperparameters. An empirical finding is that `config.sample.train_batch_size * num_gpu / config.sample.num_image_per_prompt * config.sample.num_batches_per_epoch = 48`, i.e., `group_number=48`, `group_size=24`.
Additionally, setting `config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2` also yields good performance.

## 🤗 Acknowledgement

This project is based on [Flow-GRPO](https://github.com/yifan123/flow_grpo). Thank you for your outstanding contributions to the community.
=======
# CGPO
>>>>>>> 9ea26ba890553638edd19feb62d9bbc10bef126a
