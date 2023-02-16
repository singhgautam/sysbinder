
# Neural Systematic Binder
*ICLR 2023*

#### [[arXiv](https://arxiv.org/abs/2211.01177)] [[project](https://sites.google.com/view/neural-systematic-binder)] [[datasets](https://drive.google.com/drive/folders/1FKEjZnKfu9KnSGfnr8oGVUBSPqnptzJc?usp=sharing)] [[openreview](https://openreview.net/forum?id=ZPHE4fht19t)]

This is the **official PyTorch implementation** of _Neural Systematic Binder_.

<img src="https://i.imgur.com/hqwcCpU.png">

### Authors
Gautam Singh and Yeongbin Kim and Sungjin Ahn

### Datasets
The datasets tested in the paper (CLEVR-Easy, CLEVR-Hard, and CLEVR-Tex) can be downloaded via this [link](https://drive.google.com/drive/folders/1FKEjZnKfu9KnSGfnr8oGVUBSPqnptzJc?usp=sharing).

### Training
To train the model, simply execute:
```bash
python train.py
```
Check `train.py` to see the full list of training arguments. You can use the `--data_path` argument to point to the set of images via a glob pattern.

### Outputs
The training code produces Tensorboard logs. To see these logs, run Tensorboard on the logging directory that was provided in the training argument `--log_path`. These logs contain the training loss curves and visualizations of reconstructions and object attention maps.

### Packages Required
The following packages may need to be installed first.
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://pypi.org/project/tensorboard/) for logging.

### Evaluation
The evaluation scripts are provided in branch `evaluate`.

### Citation
```
@inproceedings{
      singh2023sysbinder,
      title={Neural Systematic Binder},
      author={Gautam Singh and Yeongbin Kim and Sungjin Ahn},
      booktitle={International Conference on Learning Representations},
      year={2023},
      url={https://openreview.net/forum?id=ZPHE4fht19t}
}
```