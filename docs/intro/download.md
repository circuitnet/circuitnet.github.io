---
title: "Download"
---

# Download

Choose the appropriate dataset based on your target tasks:

| Task | Recommended Dataset |
| --- | --- |
| Congestion Prediction | Routability Features |
| DRC Violation Prediction | Routability Features |
| IR Drop Analysis | IR Drop Features |
| Net Delay Prediction | Graph Features / Timing Features |

## Available Datasets

### CircuitNet-N28

- [Google Drive](https://drive.google.com/drive/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/1evSTtuvphyl1_aSedsEQLA?pwd=wihf)

### CircuitNet-N14

- 🤗 [Hugging Face](https://huggingface.co/datasets/CircuitNet/CircuitNet/tree/main)

### CircuitNet-N45

- 🤗 [Hugging Face](https://huggingface.co/datasets/SKLP-EDA-LAB/CircuitNet3.0)

## Decompress and Preprocess

::: tip Note
The decompression process may take at least 15 minutes and require 200GB of storage space. Please ensure you have sufficient disk space before proceeding. **For the [CircuitNet-N14 on Hugging Face](https://huggingface.co/datasets/CircuitNet/CircuitNet/tree/main), the detailed decompress instructions are coming soon.**
:::

### Routability/IR Drop Features

#### Step 1: Decompress the Dataset

Choose one of the following commands based on your needs:

```bash
# For Routability features
python decompress_routability.py

# For IR Drop features
python decompress_IR_drop.py
```

::: warning Important
Make sure your directory structure matches the one in Google Drive or Baidu Netdisk, and you are using the latest version of the script from the drive.
:::

#### Step 2: Generate Training Set

Run the preprocessing script to generate the training set for your specific task:

```bash
python generate_training_set.py \
    --task [congestion/DRC/IR_drop] \
    --data_path [path_to_decompressed_dataset] \
    --save_path [path_to_save_output]
```

#### Step 3: Start Training

You can now:

- Set up your own model for training
- Use our tutorial code from the [tutorial page](https://circuitnet.github.io/tutorial/experiment_tutorial.html)
- Check our [GitHub repository](https://github.com/circuitnet/CircuitNet) for implementation details

### Graph Features

#### Step 1: Decompress the Dataset

```bash
tar -xf PATH_TO_THE_FILE
```

#### Step 2: Construct Graph

Sample code for graph construction will be available soon.

## Raw Data

We provide the following raw data formats (**CircuitNet-N28**) for custom feature extraction:

- Netlist files
- LEF/DEF files

Raw data for CircuitNet-N14 is coming soon. Currently available upon request.

### Feature Extraction

You can use our [feature extraction toolkit](https://github.com/circuitnet/CircuitNet/tree/main/feature_extraction) to:

- Extract custom features from raw data
- Process and transform the data for your specific needs
- Integrate with your own machine learning pipeline

For detailed instructions on feature extraction, please refer to our [feature documentation](/feature/properties.html).

::: warning Note
Make sure to follow the data format specifications when working with raw data files. The feature extraction toolkit includes example scripts and documentation to help you get started.
:::
