# Intro

## CircuitNet

CircuitNet is an open-source dataset dedicated to machine learning (ML) applications in electronic design automation (EDA). We have collected more than 10K samples from versatile runs of commercial design tools based on open-source RISC-V designs with various features for multiple ML for EDA applications. The features are saved seperately as below:


.               
├── **Routability_features**  

| ├── cell_density

| └── congestion

<!-- | | ├── ... -->

| | ├── congestion_early_global_routing

| | | ├── overflow_based

| | | | ├── congestion_eGR_horizontal_overflow

| | | | └── congestion_eGR_vertical_overflow

| | | └── utilization_based

| | |   ├── congestion_eGR_horizontal_util

| | |   └── congestion_eGR_vertical_util

| | └── congestion_global_routing

| |   ├── overflow_based

| |   | ├── congestion_GR_horizontal_overflow

| |   | └── congestion_GR_vertical_overflow

| |   └── utilization_based

| |     ├── congestion_GR_horizontal_util

| |     └── congestion_GR_vertical_util

| ├── drc

| | ├── drc_all

| | └── drc_seperated

| ├── macro_region

| └── RUDY

|   ├── RUDY

|   ├── long_RUDY

|   ├── short_RUDY

|   ├── pin_RUDY

|   └── long_pin_RUDY

├── **Irdrop_features**  

| ├── Power_i

| ├── Power_s

| ├── Power_sca

| ├── Power_all

| ├── Power_t

| └── ir_drop

├── doc

| └── UserGuide.pdf  

└── script  

  └── generate_training_set.py

<!-- <pre class="folder-tree">
  Routability_features
    cell_density
    congestion
      congestion_early_global_routing
        overflow_based
          congestion_eGR_horizontal_overflow
          congestion_eGR_vertical_overflow
        utilization_based
          congestion_eGR_horizontal_util
          congestion_eGR_vertical_util
      congestion_global_routing
        overflow_based
          congestion_GR_horizontal_overflow
          congestion_GR_vertical_overflow
        utilization_based
          congestion_GR_horizontal_util
          congestion_GR_vertical_util
    drc
      drc_all
      drc_seperated
    macro_region
    RUDY
      RUDY
      long_RUDY
      short_RUDY
      pin_RUDY
      long_pin_RUDY
  Irdrop_features
    Power
      Power_i
      Power_s
      Power_sca
      Power_all
      Power_t
    ir_drop
    doc
      UserGuide.pdf  
  script
    code
    generate_training_set.py
</pre> -->


  We separate the features and store them in different directories to enable custom applications. Thus they need to be preprocessed and combined in certain arrangement for training.  Our scripts can preprocess and combine different features for training and testing.  But we also encourage to implement different preprocessing methods and use different combinations of features.

<!-- To evaluate the dataset, we have implement 7 models on 3 tasks, i.e. congestion prediction, DRC violations prediction, IR drop prediction. The implemention code is also open-sourced, and we also provide script for generating traing set in these experiments so that you will be able to reproduce our results. On the other hand, you can use the script as guide for implementing your own method. -->



<!-- <script src="./folder-tree.js"></script>
  <script>
    var elements = document.getElementsByClassName('folder-tree'),
        length = elements.length;

    for (var i = length - 1; i >= 0; --i) {
      var node = elements[i],
          container = document.createElement('span');
      container.innerHTML = folderTree(node.innerHTML);

      node.parentNode.replaceChild(container.firstChild, node);
    }
  </script>
</body> -->