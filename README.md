# CircuitNet

CircuitNet is an open-source dataset dedicated to machine learning (ML) applications in electronic design automation (EDA). We have collected more than 20K samples from versatile runs of commercial design tools based on open-source designs with various features for multiple ML for EDA applications.

This documentation is organized as followed:

- [Introduction](https://circuitnet.github.io/intro/intro.html): introduction and quick start.
- [Feature Description](https://circuitnet.github.io/feature/properties.html): name conventions, calculation method, characteristics and visualization.

The codes in the [tutorial page](https://circuitnet.github.io/tutorial/experiment_tutorial.html) is available in our github repository [https://github.com/circuitnet/CircuitNet](https://github.com/circuitnet/CircuitNet).

This project is under active development. We are expanding the dataset to include diverse and large-scale designs for versatile ML applications in EDA. If you have any feedback or questions, please feel free to contact us or raise a issue in our github repository.

[Download](https://circuitnet.github.io/intro/download.html)

### Citation

[TCAD](https://ieeexplore.ieee.org/document/10158384)

```
@ARTICLE{10158384,
  author={Chai, Zhuomin and Zhao, Yuxiang and Liu, Wei and Lin, Yibo and Wang, Runsheng and Huang, Ru},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={CircuitNet: An Open-Source Dataset for Machine Learning in VLSI CAD Applications with Improved Domain-Specific Evaluation Metric and Learning Strategies}, 
  year={2023},
  doi={10.1109/TCAD.2023.3287970}}
}

```

[ICLR](https://openreview.net/forum?id=nMFSUjxMIl)

```
@inproceedings{
2024circuitnet,
title={CircuitNet 2.0: An Advanced Dataset for Promoting Machine Learning Innovations in Realistic Chip Design Environment},
author={Xun, Jiang and Chai, Zhuomin and Zhao, Yuxiang and Lin, Yibo and Wang, Runsheng and Huang, Ru},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=nMFSUjxMIl}
}

```

### Change Log
- 2023/7/24

  Code for feature extraction released. Users can use it to implement self-defined features with the LEF/DEF we released or extract features with LEF/DEF from other sources. Read the [REAME](https://github.com/circuitnet/CircuitNet/blob/main/feature_extraction/README.md) for more information.

- 2023/6/29

  Code for net delay prediction released. A simple tutorial on net delay prediction is added to [our website](https://circuitnet.github.io/tutorial/experiment_tutorial.html#Net_Delay).

- 2023/6/14

  The original dataset is renamed to CircuitNet-N28, and additional timing features are released.

  New dataset CircuitNet-N14 is released, supporting congestion, IR drop and timing prediction. 

  <!-- Codes for feature extraction and net delay prediction coming soon.  -->

- 2023/3/22 

  LEF/DEF is updated to include tech information (sanitized).

  Congestion features and graph features generated from ISPD2015 benchmark are available in the ISPD2015 dir in [Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and [Baidu Netdisk](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fapps%2Fbypy%2FCircuitNet).

- 2022/12/29 

  LEF/DEF (sanitized) are available in the LEF&DEF dir in [Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and [Baidu Netdisk](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fapps%2Fbypy%2FCircuitNet).

- 2022/12/12 
  
  Graph features are available in the graph_features dir in [Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and [Baidu Netdisk](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fapps%2Fbypy%2FCircuitNet).

- 2022/9/6 

  Pretrained weights are available in [Google Drive](https://drive.google.com/drive/folders/10PD4zNa9fiVeBDQ0-drBwZ3TDEjQ3gmf?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1dUEt35PQssS7_V4fRHwWTQ?pwd=7i67).

- 2022/8/1 
  
  First release.
