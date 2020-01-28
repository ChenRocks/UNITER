# UNITER: Learning UNiversal Image-TExt Representations
This is the official repository of [UNITER](https://arxiv.org/abs/1909.11740).
It is currently an alpha release, which supports finetuning UNITER-base on the
[NLVR2](http://lil.nlp.cornell.edu/nlvr/) task.
We plan to release the large model and more downstream tasks but do not have a 
time table as of now.

![Overview of UNITER](https://convaisharables.blob.core.windows.net/uniter/uniter_overview.png)

Some code in this repo are copied/modified from opensource implementations made available by
[PyTorch](https://github.com/pytorch/pytorch),
[HuggingFace](https://github.com/huggingface/transformers),
[OpenNMT](https://github.com/OpenNMT/OpenNMT-py),
and [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch).
The image features are extracted using [BUTD](https://github.com/peteanderson80/bottom-up-attention).


## Requirements
We provide Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.

## Quick Start
1. Download processed data and pretrained models with the following command.
```bash
bash scripts/download.sh $PATH_TO_STORAGE
```
After downloading you should see the following folder structure:
```
├── ann
│   ├── dev.json
│   └── test1.json
├── finetune
│   ├── nlvr-base
│   └── nlvr-base.tar
├── img_db
│   ├── nlvr2_dev
│   ├── nlvr2_dev.tar
│   ├── nlvr2_test
│   ├── nlvr2_test.tar
│   ├── nlvr2_train
│   └── nlvr2_train.tar
├── pretrained
│   └── uniter-base.pt
└── txt_db
    ├── nlvr2_dev.db
    ├── nlvr2_dev.db.tar
    ├── nlvr2_test1.db
    ├── nlvr2_test1.db.tar
    ├── nlvr2_train.db
    └── nlvr2_train.db.tar
```

2. Launch the Docker container for running the experiments.
```bash
# docker image should be automatically pulled
source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/img_db \
    $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
```
The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
Note that the source code is mounted into the container under `/src` instead 
of built into the image so that user modification will be reflected without
re-building the image. (Data folders are mounted into the container separately
for flexibility on folder structures.)


3. Run finetuning for the NLVR2 task.
```bash
# inside the container
python train_nlvr2.py --config config/train-nlvr2-base-1gpu.json

# for more customization
horovodrun -np $N_GPU python train_nlvr2.py --config $YOUR_CONFIG_JSON
```

4. Run inference for the NLVR2 task and then evaluate.
```bash
# inference
python inf_nlvr2.py --txt_db /txt/nlvr2_test1.db/ --img_db /img/nlvr2_test/ \
	--train_dir /storage/nlvr-base/ --ckpt 6500 --output_dir . --fp16

# evaluation
# run this command outside docker (tested with python 3.6)
# or copy the annotation json into mounted folder
python scripts/eval_nlvr2.py ./results.csv $PATH_TO_STORAGE/ann/test1.json
```
The above command runs inference on the model we trained. Feel free to replace
`--train_dir` and `--ckpt` with your own model trained in step 3.
Currently we only support single GPU inference.


5. Customization
```bash
# training options
python train_nlvr2.py --help
```
- command-line argument overwrites JSON config files
- JSON config overwrites `argparse` default value.
- use horovodrun to run multi-GPU training
- `--gradient_accumulation_steps` emulates multi-gpu training


6. Misc.
```bash
# text annotation preprocessing
bash scripts/create_txtdb.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/ann

# image feature extraction (Tested on Titan-Xp; may not run on latest GPUs)
bash scripts/extract_imgfeat.sh $PATH_TO_IMG_FOLDER $PATH_TO_IMG_NPY

# image preprocessing
bash scripts/create_imgdb.sh $PATH_TO_IMG_NPY $PATH_TO_STORAGE/img_db
```
In case you would like to reproduce the whole preprocessing pipeline.


## Citation

If you find this code useful for your research, please consider citing:
```
@article{chen2019uniter,
  title={Uniter: Learning universal image-text representations},
  author={Chen, Yen-Chun and Li, Linjie and Yu, Licheng and Kholy, Ahmed El and Ahmed, Faisal and Gan, Zhe and Cheng, Yu and Liu, Jingjing},
  journal={arXiv preprint arXiv:1909.11740},
  year={2019}
}
```

## License

MIT
