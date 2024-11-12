# Timing Features

Timing prediction includes net delay prediction and endpoint slack prediction, and we supports net delay prediction for now.

## Net Delay Prediction

Features for net delay prediction is pin positions on the chip in micron, which is extracted from DEF, and the label is the net delay in nanosecond in 4 fabrication corners on the chip (early/late & rise/fall, and if you don't understand this, just take them as 4 parallel prediction tasks), which is extracted from the SDF (Standard Delay Format) file.

**For ease of use, we extract the necessary information for forming graph.**
To be specific, a pin-oriented graph can be built, where pins of the instances are nodes and the nets that connecting the pins are edges, then pin positions can be node features to predict net delays which are edge features.

For CircuitNet-N28, the features are extracted after routing. For CircuitNet-N14, the features are extracted 3 times, after placement, after CTS (clock tree synthesis) and after routing, respectively. They can all be used in net delay prediction.

### net_edges

They are saved in the form of npz (a zipped archive of numpy array) in net_edges.tar.gz.
You can load the npz file with:

`numpy.load(FILE_NAME)['net_edges']` 

The shape of the numpy array is [#edges, edge_features], and there are 6 features, which are the 6 channels in the net_edges array:

- [0]: the **source node** index of the edge.
- [1]: the **destination node** index of the edge.
- [2]: the net delay in early and rise corner.
- [3]: the net delay in late and rise corner.
- [4]: the net delay in early and fall corner.
- [5]: the net delay in late and fall corner.

so for a design with 1000 nets, the array will be [1000,6].
And with channel 0 and channel 1 (source nodes index and destination nodes index), you can directly built a graph with dgl.

### pin_positions

Then the node features are pin positions, and they are saved as a npz file containing a dictionary in pin_positions.tar.gz.
The dictionary can be loaded with

`numpy.load(FILE_NAME, allow_pickle=True)['pin_positions'].item()` 

The dictionary has pin names as keys, and a list as value. The list contains 2 features, which are pin positions in micron and pin positions on gcell grid. The pin positions have 4 values, which are the left

### nodes

To access the pin_positions dictionary, we need pin names (pins are nodes in the graph) as keys, and the mapping from pin index to pin names are saved in a npz file in nodes.tar.gz.
The array can be accessed with

`numpy.load(FILE_NAME)['nodes']`

It is a 1 dimension array, and the node index in net_edges array can be used to access the corresponding node name through indexing. Then the node name can be used to access the pin_positions dictionary to get pin position as node feature. 
