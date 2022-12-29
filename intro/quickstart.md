# Quick Start

(1)Based on your target tasks, download Routability Features(for congestion and DRC) or IR Drop Features(for IR drop).

[Google Drive](https://drive.google.com/drive/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB?usp=sharing)

[Baidu Netdisk](https://pan.baidu.com/s/1evSTtuvphyl1_aSedsEQLA?pwd=wihf)

Decompress with scripts in the script dir

`python decompress_routability.py`

or 

`python decompress_IR_drop.py`

Make sure your directory structure is the same as the one in Google Drive or Baidu Netdisk and you are using the latest version of the script in the drive.

This may takes at least 15 minutes and 200G storage space, please be patient.

(2)Run preprocessing script to generate training set for corresponding tasks. Specify your task with option: congestion/DRC/IR_drop.

```python generate_training_set.py --task [congestion/DRC/IR_drop] --data_path [path_to_decompressed_dataset] --save_path [path_to_save_output]```

(3)Now, you can set up your own model for training or use the tutorial code from our [tutorial page](https://circuitnet.github.io/tutorial/experiment_tutorial5.html) and github repository [https://github.com/circuitnet/CircuitNet](https://github.com/circuitnet/CircuitNet). If you meet any problems, feel free to open an issue in our repository or contact us by email.

