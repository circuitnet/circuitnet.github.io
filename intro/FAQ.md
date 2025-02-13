# FAQ

1. How to visualize the features? Why my visualization is different from DEF.

    You can use [this script](https://github.com/circuitnet/CircuitNet/blob/main/feature_extraction/vis.py) as the template to visualize the features.
    Note that you need to rotate the image 90 degrees clockwise (like np.rot90) due to the difference between the coordinate system of numpy array and image.

2. How to add new feature with the dataset?

    We have released the raw data (LEF&DEF, netlist) for CircuitNet-N28/ISPD2015 and the scripts for feature extraction. You can modify the scripts to add your own feature.

3. Why the released LEF&DEF cannot be loaded in OpenROAD or Innovus?

    The main purpose of releasing LEF&DEF is to support user-defined feature extraction, and they are not meant to be used for running flows.
    Due to the limitation of NDA, we sanitize the LEF&DEF in our release, and some information has been omitted, leading to possible problems in OpenROAD and Innovus. It is possible to use them in simpler placer or router (like DREAMPlace), but there is no gurantee. 

4. The number of instances are different in graph_information and instance_placement.

    The graph_information is extracted with the netlist after logic synthesis and the instance_placement is extracted with the DEF after placement. We use Innovus for PnR, and Innovus will delete or add instances during this process. We recommend to simply take the intersection.

5. Have errors during decompression or preprocessing.

    For decompression, please check that you have the same directory structure as the one in our netdisk, then use our script for decompression.
    For preprocessing, please check the paths in the error do exist.