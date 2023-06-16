# CircuitNet

CircuitNet is an open-source dataset dedicated to machine learning (ML) applications in electronic design automation (EDA). We have collected more than 10K samples from versatile runs of commercial design tools based on open-source RISC-V designs with various features for multiple ML for EDA applications.

This documentation is organized as followed:

- [Introduction](https://circuitnet.github.io/intro/intro.html): introduction and quick start.
- [Feature Description](https://circuitnet.github.io/feature/properties.html): name conventions, calculation method, characteristics and visualization.

The codes in the [tutorial page](https://circuitnet.github.io/tutorial/experiment_tutorial.html) is available in our github repository [https://github.com/circuitnet/CircuitNet](https://github.com/circuitnet/CircuitNet).

This project is under active development. We are expanding the dataset to include diverse and large-scale designs for versatile ML applications in EDA. If you have any feedback or questions, please feel free to contact us.

[Download](https://circuitnet.github.io/intro/download.html)

### Citation

[Paper Link](https://www.sciengine.com/SCIS/doi/10.1007/s11432-022-3571-8)

```
@article{chai2022circuitnet,
  title = {CircuitNet: An Open-Source Dataset for Machine Learning Applications in Electronic Design Automation (EDA)},
  author = {Chai, Zhuomin and Zhao, Yuxiang and Lin, Yibo and Liu, Wei and Wang, Runsheng and Huang, Ru},
  journal= {SCIENCE CHINA Information Sciences},
  volume={65},
  number = "12",
  pages={227401-},
  year = {2022}
}

```

### Change Log
- 2022/8/1 
  
  First release.
- 2022/9/6 

  Pretrained weights are available in [Google Drive](https://drive.google.com/drive/folders/10PD4zNa9fiVeBDQ0-drBwZ3TDEjQ3gmf?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1dUEt35PQssS7_V4fRHwWTQ?pwd=7i67).
- 2022/12/12 
  
  Graph features are available in the graph_features dir in [Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and [Baidu Netdisk](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fapps%2Fbypy%2FCircuitNet).
- 2022/12/29 

  LEF/DEF (sanitized) are available in the LEF&DEF dir in [Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and [Baidu Netdisk](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fapps%2Fbypy%2FCircuitNet).

- 2023/3/22 

  LEF/DEF is updated to include tech information (sanitized).

  Congestion features and graph features generated from ISPD2015 benchmark are available in the ISPD2015 dir in [Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and [Baidu Netdisk](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fapps%2Fbypy%2FCircuitNet).

- 2023/6/14

  The original dataset is renamed to CircuitNet-N28, and timing features are released.

  New dataset CircuitNet-N14 is released, supporting congestion, IR drop and net delay prediction. 

  Codes for feature extraction and net delay prediction coming soon. 