## Experiments&Benchmarks

------

Here, we select several representative methods to give a brief introduction to applying machine learning to VLSI physical design cycle that provides an intuitive awareness of the functionality and practicability of `CircuirNet` to users.

### Congestion Prediction

The network of `Global Placement with Deep Learning-Enabled Explicit Routability Optimization` uses an encoder and decoder architecture to translate the image-like features into a routing resource assumption heat map (congestion map). The architecture is shown below.

<div align="center">
  <img src="../pics/1.png"  width="600">
</div>

Three image-like features of RUDY, PinRUDY and MacroRegion were fed into the network to get the final congestion prediction. Here is the visualization of input features.

<div align="center">
  <img src="../pics/2.png"  width="300">
</div>

We train the network in an end-to-end manner and compute the loss between the output and the golden result obtained by Innovus global router. The visualization of output image is shown below after training convergence.

<div align="center">
  <img src="../pics/3.png"  width="300">
</div>

### DRC Violation

DRC Violation prediction is an essential step in the physical design procedure aiming at detecting violation hotspots at the early design stage, which is quite conducive to reducing the chip design turn-around. `RouteNet: Routability Prediction for Mixed-Size Designs Using Convolutional Neural Network` is a typical method for accurately detecting violation hotspots.

<div align="center">
  <img src="../pics/4.png"  width="600">
</div>

Nine features extracted at different stages of physical design flow are combined together as one input tensor.

<div align="center">
  <img src="../pics/5.png"  width="300">
</div>

<!-- <div align="center">
  <img src="../pics/6.png" >
</div> -->


After finishing the training phase, the prediction map can be specially demonstrated into a binary matrix, where the area greater than zero depicts the potential DRC violation in designing space.



<div align="center">
  <img src="../pics/7.png" width="300">
</div>


ROC and PRC are also provided to measure the performance of the abovementioned method.

<div align="center">
  <img src="../pics/8.png" title="" alt="" width="281"> <img src="../pics/9.png" title="" alt="" width="291">
</div>

### IR Drop

IR Drop is another critical part of the whole design workflow that hugely affects the timing frequency and availability that needed to be carefully considered. `MAVIREC: ML-Aided Vectored IR-Drop Estimation and Classification` also cast the IR Drop prediction problem as an image-to-image translation task. Due to the demand for joint perception along the temporal and spatial axis, MAVIREC introduces a 3D encoder to aggregate the Spatio-temporal features and decode the prediction result into a 2D hotspot map.


<div align="center">
  <img src="../pics/10.png"  width="600">
</div>

Here is the visualization of input features.

<div align="center">
  <img src="../pics/11.png" width="300">
</div>

The training phase is stopped after the network is sufficiently capable to generate a high-quality prediction map. We also use a binary map to indicate IR Drop hotspot.

<div align="center">
  <img src="../pics/12.png" width="300">
</div>


ROC and PRC are used as assessment indices to evaluate prediction results.

<div align="center">
  <img src="../pics/13.png" width="278" > <img src="../pics/14.png" width="278">
</div>

