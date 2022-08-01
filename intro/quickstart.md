# Quick Start

(1)Based on your target tasks, download Routability Features(for congestion and DRC) or IR Drop Features(for IR drop).

[Google Drive](https://drive.google.com/drive/folders/1zB002-Aq7OcW0NgiQbkS-GBdmn6hrmZM?usp=sharing)

[Baidu Netdisk](https://pan.baidu.com/s/1hZLH22b7LLHYg_ECbdHnJA?pwd=1yvz)

Decompress with scripts in the script dir

`python decompress_routability.py`

or 

`python decompress_IR_drop.py`

This may take sometime, please be patient.

(2)Run preprocessing script to generate training set for coressponding tasks. Specify your task with option: congestion/DRC/IR_drop.

`python generate_training_set.py --task [congestion/DRC/IR_drop] --data_path [path_to_decompressed_dataset] --save_path [path_to_save_output]`

