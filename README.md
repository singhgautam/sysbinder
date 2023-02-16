
# Neural Systematic Binder
*ICLR 2023*

#### [[arXiv](https://arxiv.org/abs/2211.01177)] [[project](https://sites.google.com/view/neural-systematic-binder)] [[datasets](https://drive.google.com/drive/folders/1FKEjZnKfu9KnSGfnr8oGVUBSPqnptzJc?usp=sharing)] [[openreview](https://openreview.net/forum?id=ZPHE4fht19t)]

This is the **official PyTorch implementation** of _Neural Systematic Binder_. In this branch, we provide the evaluation scripts.

<img src="https://i.imgur.com/hqwcCpU.png">

### Authors
Gautam Singh and Yeongbin Kim and Sungjin Ahn

### Datasets
The datasets tested in the paper (CLEVR-Easy, CLEVR-Hard, and CLEVR-Tex) can be downloaded via this [link](https://drive.google.com/drive/folders/1FKEjZnKfu9KnSGfnr8oGVUBSPqnptzJc?usp=sharing). We provide the code for loading the test split of CLEVR-Easy with annotations such as masks and factor labels.

### Evaluate
To evaluate the model, simply execute:
```bash
python evaluate.py --data_path "test_data/*.png" --load_path "model.pt"
```
Check `evaluate.py` to see the full list of arguments. Using `--data_path` argument to point to the set of images via a glob pattern. Using `--load_path`, you can provide the path to the model that you want to evaluate.

### Outputs
The evaluation code produces prints the Disentanglement, Completeness and the Informativeness (DCI) scores.

### Packages Required
The following packages may need to be installed first.
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://pypi.org/project/tensorboard/) for logging.

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