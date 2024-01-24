# Graph Features

To enable application for graph based methods, we further provide 54 gate-level netlists and 10242 instance placement information.

## Gate-level Netlist

The gate-level netlists are the ones used in data generation. They are synthesised from 6 RISC-V designs with commercial 28 nm library and Synopsys Design Compiler(DC) with multiple variations. (see page [Feature](https://circuitnet.github.io/feature/properties.html) for detailed information about variations) The name of standard cell and IP is encrypted due to copyright issue. 

There are 2 version of netlist, one is the original hierarchical version written by DC, and the other one is the flatten version written by Innovus for extracting graph.

**For ease of use, we extract the necessary information for forming graph from the flatten netlist.** They are saved in the form of numpy array in graph_information.tar.gz.

(1) **Pin Attributes**: pin names, corresponding net index, node index.

(2) **Net Attributes**: net name.

(3) **Node Attributes**: node(instance) name, corresponding standard cell / IP name.

The array can be loaded with

`numpy.load(FILE_NAME, allow_pickle=True)`

Here is a simple example:

<div align="center">
  <img src="../pics/netlist.jpg" width="300">
</div>

Pin Attributes: [[I,ZN,I,ZN], [0,1,1,2], [0,0,1,1]]

Net Attributes: [[in, n1, out]]

Node Attributes：[[inv_1, inv_2],[INV,INV]]

The pins with same net/node index belong to the same net/node, e.g., pin[1] "ZN" and pin[2] "I" have net index "1", both of them belong to net "n1"; pin[0] and pin[1] have node index "0", both of them belong to node "inv_1". Thus, an adjacency matrix can be formulated through traversing the pin attributes array.

## Instance Placement (Cell Position)

The instance, i.e., standard cell and IP, is placed at certain location on layout after placement stage in back-end design. 

The placement information for each layout, i.e., the location of instances, is extracted from def and saved as a dictionary in instance_placement.tar.gz. It contains the name of instance (consistent with the ones in netlist) and the coordination for the bounding box of instance on layout. 

e.g., InstanceN : [left, bottom, right, top]

For "instance_placement_gcell", the coordinate is the same as the one in the image-like feature, which is in GCell grids, and the value indicates which GCell the instance is in.

For "instance_placement_micron", the coordinate is in micron, and the value indicates the . It can be converted to GCell grids through dividing the width of GCell (available in the corresponding DEF files).


The dictionary can be loaded with

`numpy.load(FILE_NAME, allow_pickle=True).item()`

The placement information can be added into node attributes through indexing, since there is partially one-to-one correspondence from the node name from the node attribute array above to the keys in the instance placement dictionary (they are not exactly the same because innovus will add or delete cells, just take the intersection). With the additional placement information, tasks like congestion prediction can be completed with graph neural network.

