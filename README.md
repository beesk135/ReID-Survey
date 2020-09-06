# ReID-Survey with a Powerful AGW Baseline
Deep Learning for Person Re-identification:  A Survey and Outlook. [arXiv](https://arxiv.org/abs/2001.04193)

An implementation of AGW for cross-modality visible-infrared Re-ID is [HERE](https://github.com/mangye16/Cross-Modal-Re-ID-baseline).


## Highlights

- A comprehensive survey with in-depth analysis for person Re-ID in recent years (2016-2019).

- A new evaluation metric, namely mean Inverse Negative Penalty (mINP), which measures the ability to find the hardest correct match.

- A new AGW baseline with non-local Attention block, Generalized mean pooling and Weighted regularization triplet. It acheieves competitive performance on both single-modality and [cross-modality](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) Re-ID tasks.


## AGW on Single-Modality Re-ID with mINP

### DukeMTMC dataset

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------------------------|
|BagTricks     | ImageNet | 86.4% | 76.4%|  40.7% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [PDF](https://arxiv.org/abs/1903.07071)|
|ABD-Net     | ImageNet | 89.0% | 78.6%| 42.1% | [Code](https://github.com/TAMU-VITA/ABD-Net) |ABD-Net: Attentive but Diverse Person Re-Identification. In ICCV 19. [PDF](https://arxiv.org/abs/1908.01114)|
|AGW     | ImageNet | 89.0%  | 79.6% | 45.7% | [GoogleDrive](https://drive.google.com/open?id=1q3n_acTe-vaEeIpkJG2k0HqSEZrTJoGA)| Deep Learning for Person Re-identification:  A Survey and Outlook  |

### Market-1501 dataset

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------|
|BagTricks     | ImageNet | 94.5% | 85.9%|  59.4% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [arXiv](https://arxiv.org/abs/1903.07071)|
|ABD-Net     | ImageNet | 95.6% | 88.3%|  66.2% | [Code](https://github.com/TAMU-VITA/ABD-Net) |ABD-Net: Attentive but Diverse Person Re-Identification. In ICCV 19. [PDF](https://arxiv.org/abs/1908.01114)|
|AGW     | ImageNet | 95.1%  | 87.8% | 65.0% | [GoogleDrive](https://drive.google.com/open?id=1Ymt2q3k0uBpaw5hCVscl0a29uKI1cRPA)| Deep Learning for Person Re-identification:  A Survey and Outlook. In ArXiv 20. [arXiv](https://arxiv.org/abs/2001.04193) |


### CUHK03 dataset

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------|
|BagTricks     | ImageNet | 58.0% | 56.6%|  43.8% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [PDF](https://arxiv.org/abs/1903.07071)|
|AGW     | ImageNet | 63.6%  | 62.0% | 50.3% | [GoogleDrive](https://drive.google.com/open?id=1Uyq_JBM2N1JL-buYWkLZFMd7N-eMjOUZ)| Deep Learning for Person Re-identification:  A Survey and Outlook. In ArXiv 20. [arXiv](https://arxiv.org/abs/2001.04193)   |

### MSMT17 dataset

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------|
|BagTricks     | ImageNet | 63.4% | 45.1%|  12.4% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [arXiv](https://arxiv.org/abs/1903.07071)|
|AGW     | ImageNet | 68.3% | 49.3%|  14.7% | [GoogleDrive](https://drive.google.com/open?id=1xw-t7gVkEghkgHai0nL28VhpS7mBHNG8)| Deep Learning for Person Re-identification:  A Survey and Outlook. In ArXiv 20. [arXiv](https://arxiv.org/abs/2001.04193)   |

## Quick Start

### 1. Prepare dataset 
Create a directory to store reid datasets under this repo, taking Market1501 for example
```
cd ReID-Survey
mkdir toDataset
```
- Set ```_C.DATASETS.ROOT_DIR = ('./toDataset')``` in```config/defaults.py```
- Download dataset to toDataset/ from [http://www.liangzheng.org/Project/project_reid.html](http://www.liangzheng.org/Project/project_reid.html)

- Extract dataset and rename to ```market1501```. The data structure would like:
```
toDataset
    market1501 
        bounding_box_test/
        bounding_box_train/
        ......
```

### 2. Install dependencies

  - pytorch>=0.4
  - torchvision
  - ignite=0.1.2
  - yacs
  
### 3. Train

To train a AGW model with on Market1501 with GPU device 0, run similarly:
```
python3 tools/main.py --config_file='configs/AGW_baseline.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('./log/market1501/Experiment-AGW-baseline')"
```

### 4. Test

To test a AGW model with on Market1501 with weight file ```'./pretrained/dukemtmc_AGW.pth'```, run similarly:
```
python3 tools/main.py --config_file='configs/AGW_baseline.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')"  MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('./pretrained/market1501_AGW.pth')" TEST.EVALUATE_ONLY "('on')" OUTPUT_DIR "('./log/Test')"
```

## Citation

Please kindly cite this paper in your publications if it helps your research:
```
@article{arxiv20reidsurvey,
  title={Deep Learning for Person Re-identification: A Survey and Outlook},
  author={Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven C. H.},
  journal={arXiv preprint arXiv:2001.04193},
  year={2020},
}
```

Contact: mangye16@gmail.com
