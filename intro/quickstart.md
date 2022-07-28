# Quick Start

(1)Based on your target tasks, download Routability Features(for congestion and DRC) or IR Drop Features(for IR drop).

[Google Drive](https://drive.google.com/drive/folders/1zB002-Aq7OcW0NgiQbkS-GBdmn6hrmZM?usp=sharing)

[Baidu Netdisk](https://pan.baidu.com/s/1nCnstpG8OuOvTlStaP-3Zg?pwd=5t99)

Decompress with

`cat Routability_features.tar.gz. > Routability_features.tar.gz`

`tar -xzvf Routability_features.tar.gz`

or 
`cat Ir_drop_features.tar.gz. > Ir_drop_features.tar.gz `

`tar -xzvf Ir_drop_features.tar.gz`


(2)Run preprocessing script to generate training set for coressponding tasks. Specify your task with option: congestion/drc/irdrop.

`python generate_training_set.py $task --data_path [path_to_decompressed_dataset] --save_path [path_to_save_output]`

