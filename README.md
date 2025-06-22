# I-JEPA

Official PyTorch codebase for I-JEPA (the **Image-based Joint-Embedding Predictive Architecture**) published @ CVPR-23.
[\[arXiv\]](https://arxiv.org/pdf/2301.08243.pdf) [\[JEPAs\]](https://ai.facebook.com/blog/yann-lecun-advances-in-ai-research/) [\[blogpost\]](https://ai.facebook.com/blog/yann-lecun-ai-model-i-jepa/)



***This fork contains downstream tasks using I-JEPA models:***

* [Image classification](#Image-Classification)
* [Semantic segmentation](#Semantic-Segmentation)
* Image similarity and search

## Updates

* June 22, 2025: Added Linear Image Classification task and Semantic Segmentation task.

## Downloading Weights

Before running any training, [download the weights from here](https://github.com/facebookresearch/ijepa?tab=readme-ov-file#pretrained-models) and put them in the `weights` directory.

## Image Classification

Check `src/img_cls` folder for all the coding details.

The `train_classifier.py` in the project root directory is the executable script to start the training process.

* Steps to train:

```
python train_classifier.py --train-dir path/to/directory/with/training/class/folder --valid-dir path/to/directory/with/validation/class/folder --epochs <num_epochs>
```

```
python train_classifier.py --help
usage: train_classifier.py [-h] [-e EPOCHS] [-lr LEARNING_RATE] [-b BATCH_SIZE] [--save-name SAVE_NAME] [--fine-tune] [--out-dir OUT_DIR] [--scheduler SCHEDULER [SCHEDULER ...]]
                           --train-dir TRAIN_DIR --valid-dir VALID_DIR

options:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train our network for
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate for training the model
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  --save-name SAVE_NAME
                        file name of the final model to save
  --fine-tune           whether to fine-tune the model or train the classifier layer only
  --out-dir OUT_DIR     output sub-directory path inside the `outputs` directory
  --scheduler SCHEDULER [SCHEDULER ...]
                        number of epochs after which learning rate scheduler is applied
  --train-dir TRAIN_DIR
                        path to the training directory containing class folders in PyTorch ImageFolder format
  --valid-dir VALID_DIR
                        path to the validation directory containing class folders in PyTorch ImageFolder format
```

* Step to run image inference:

```
python infer_classifier.py --weights <path to the weights.pth file> --input <directory containing inference images>
```

## Semantic Segmentation

* Training example command:

```
python train_segmentation.py --train-images voc_2012_segmentation_data/train_images --train-masks voc_2012_segmentation_data/train_labels --valid-images voc_2012_segmentation_data/valid_images --valid-masks voc_2012_segmentation_data/valid_labels --config segmentation_configs/voc.yaml
```

Check the `segmentation_configs` directory to know more about setting up the configuration YAML files.

Check [this dataset on Kaggle](https://www.kaggle.com/datasets/sovitrath/voc-2012-segmentation-data) to know how the images and masks are structured.

* Image inference using fine-tuned model (use the same configuration YAML file as used during training for the same weights. For example for the above training, we should use `voc.yaml` during inference also.):

```
python infer_seg_image.py --input <directory/with/images> --model <best_iou_weights.pth> --config <dataset/config.yaml>
```

* Video inference using fine-tuned model (use the same configuration YAML file as used during training for the same weights. For example for the above training, we should use `voc.yaml` during inference also.):

```
python infer_seg_video.py --input <path/to/video.mp4> --model <best_iou_weights.pth> --config <dataset/config.yaml>
```

## Method
I-JEPA is a method for self-supervised learning. At a high level, I-JEPA predicts the representations of part of an image from the representations of other parts of the same image. Notably, this approach learns semantic image features:
1. without relying on pre-specified invariances to hand-crafted data transformations, which tend to be biased for particular downstream tasks,
2. and without having the model fill in pixel-level details, which tend to result in learning less semantically meaningful representations.

![ijepa](https://github.com/facebookresearch/ijepa/assets/7530871/dbad94ab-ac35-433b-8b4c-ca227886d311)

## Visualizations

As opposed to generative methods that have a pixel decoder, I-JEPA has a predictor that makes predictions in latent space.
The predictor in I-JEPA can be seen as a primitive (and restricted) world-model that is able to model spatial uncertainty in a static image from a partially observable context.
This world model is semantic in the sense that it predicts high level information about unseen regions in the image, rather than pixel-level details.

We trained a stochastic decoder that maps the I-JEPA predicted representations back in pixel space as sketches.
The model correctly captures positional uncertainty and produces high-level object parts with the correct pose (e.g., dog’s head, wolf’s front legs).

![ijepa-predictor-sketch](https://github.com/facebookresearch/ijepa/assets/7530871/9b66e461-fc8b-4b12-9f06-63ec4dfc1452)
<sub>
Caption: Illustrating how the predictor learns to model the semantics of the world. For each image, the portion outside of the blue box is encoded and given to the predictor as context. The predictor outputs a representation for what it expects to be in the region within the blue box. To visualize the prediction, we train a generative model that produces a sketch of the contents represented by the predictor output, and we show a sample output within the blue box. The predictor recognizes the semantics of what parts should be filled in (the top of the dog’s head, the bird’s leg, the wolf’s legs, the other side of the building).
</sub>

## Evaluations

I-JEPA pretraining is also computationally efficient.
It does not involve any overhead associated with applying more computationally intensive data augmentations to produce multiple views.
Only one view of the image needs to be processed by the target encoder, and only the context blocks need to be processed by the context encoder.
Empirically, I-JEPA learns strong off-the-shelf semantic representations without the use of hand-crafted view augmentations.

![1percenteval](https://github.com/facebookresearch/ijepa/assets/7530871/e6e5291f-ca51-43a4-a6cf-069811094ece)
![lineareval](https://github.com/facebookresearch/ijepa/assets/7530871/d8cffa73-5350-444e-987a-7e131a86d767)


## Pretrained models

<table>
  <tr>
    <th colspan="1">arch.</th>
    <th colspan="1">patch size</th>
    <th colspan="1">resolution</th>
    <th colspan="1">epochs</th>
    <th colspan="1">data</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td>ViT-H</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>300</td>
    <td>ImageNet-1K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-H</td>
    <td>16x16</td>
    <td>448x448</td>
    <td>300</td>
    <td>ImageNet-1K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16.448-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith16-448_ep300.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-H</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>66</td>
    <td>ImageNet-22K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in22k_vith14_ep66.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-g</td>
    <td>16x16</td>
    <td>224x224</td>
    <td>44</td>
    <td>ImageNet-22K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in22k_vitg16_ep44.yaml">configs</a></td>
  </tr>
</table>

## Code Structure

```
.
├── configs                   # directory in which all experiment '.yaml' configs are stored
├── src                       # the package
│   ├── train.py              #   the I-JEPA training loop
│   ├── helper.py             #   helper functions for init of models & opt/loading checkpoint
│   ├── transforms.py         #   pre-train data transforms
│   ├── datasets              #   datasets, data loaders, ...
│   ├── models                #   model definitions
│   ├── masks                 #   mask collators, masking utilities, ...
│   └── utils                 #   shared utilities
├── main_distributed.py       # entrypoint for launch distributed I-JEPA pretraining on SLURM cluster
└── main.py                   # entrypoint for launch I-JEPA pretraining locally on your machine
```

**Config files:**
Note that all experiment parameters are specified in config files (as opposed to command-line-arguments). See the [configs/](configs/) directory for example config files.

## Launching I-JEPA pretraining

### Single-GPU training
This implementation starts from the [main.py](main.py), which parses the experiment config file and runs the pre-training locally on a multi-GPU (or single-GPU) machine. For example, to run I-JEPA pretraining on GPUs "0","1", and "2" on a local machine using the config [configs/in1k_vith14_ep300.yaml](configs/in1k_vith14_ep300.yaml), type the command:
```
python main.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --devices cuda:0 cuda:1 cuda:2
```
*Note: This example is just used for illustrative purposes, as the ViT-H/14 config should be run on 16 A100 80G GPUs for an effective batch-size of 2048, in order to reproduce our results.*

### Multi-GPU training
In the multi-GPU setting, the implementation starts from [main_distributed.py](main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster.

For example, to pre-train on 16 A100 80G GPUs using the pre-training experiment configs specificed inside [configs/in1k_vith14_ep300.yaml](configs/in1k_vith14_ep300.yaml), type the command:
```
python main_distributed.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --folder $path_to_save_submitit_logs \
  --partition $slurm_partition \
  --nodes 2 --tasks-per-node 8 \
  --time 1000
```

---

### Requirements
* Python 3.8 (or newer)
* PyTorch 2.0
* torchvision
* Other dependencies: pyyaml, numpy, opencv, submitit

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
