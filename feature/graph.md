# Graph Features

## Gate-level Netlist & Instance Placement

To enable application for graph based methods, we further provide 54 gate-level netlists and 10242 instance placement information.

(1) The gate-level netlists are the ones used in data generation. They are synthesised from 6 RISC-V designs with commercial 28 nm library and Synopsys Design Compiler with multiple variations. (see page [Feature](https://circuitnet.github.io/feature/properties.html) for detailed information about variations) 

The name of standard cell and IP is encrypted because of copyright issue.

(2) The instance, i.e., standard cell and IP, is placed at certain location on layout after placement stage in back-end design. 

The placement information for each layout, i.e., the location of instances, is saved as a dictionary, containing the name of instance (consistent with the ones in netlist) and the coordination for the bounding box of instance on layout. 

e.g., InstanceN : [left, bottom, right, top]

The dictionary can be loaded with

`numpy.load(FILE_NAME, allow_pickle=True)`

(3) Graph can be obtained with the connectivity information from netlist as edges, and the instance placement information as vertices.