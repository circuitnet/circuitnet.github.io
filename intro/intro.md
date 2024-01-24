# Intro

## CircuitNet

CircuitNet is an open-source dataset dedicated to machine learning (ML) applications in electronic design automation (EDA). We have collected more than 20K samples from versatile runs of commercial design tools based on open-source designs with various features for multiple ML for EDA applications. 

We now have 3 variations of datasets, **CircuitNet-N28**, **CircuitNet-N14** and **ISPD2015**. They are all collected from runs of commercial design tools, but they are based on different designs and technology. 

**CircuitNet-N28** is based on RISC-V designs and 28nm planar technology, and now it provides the most comprehensive support for all tasks.

**CircuitNet-N14** is the advanced version of CircuitNet-N28. It includes more designs other than RISC-V, including GPU and ML accelerator, and based on 14nm FinFET technology. This dataset only supports congestion prediction, IR drop prediction and net delay prediction, but will extends to DRC prediction in the future.

**CircuitNet-ISPD15** is based on the [ISPD2015 contest benchmark](https://www.ispd.cc/contests/15/ispd2015contest.html). Due to the original purpose of the benchmark, this dataset only supports congestion prediction, but will extends to DRC prediction in the future.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax">Congestion</th>
    <th class="tg-0lax">DRC</th>
    <th class="tg-0lax">IR drop</th>
    <th class="tg-0lax">Net Delay</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">CircuitNet-N28</td>
    <td class="tg-0lax">yes</td>
    <td class="tg-0lax">yes</td>
    <td class="tg-0lax">yes</td>
    <td class="tg-0lax">yes</td>
  </tr>
  <tr>
    <td class="tg-0lax">CircuitNet-N14</td>
    <td class="tg-0lax">yes</td>
    <td class="tg-0lax">no</td>
    <td class="tg-0lax">yes</td>
    <td class="tg-0lax">yes</td>
  </tr>
  <tr>
    <td class="tg-0lax">CircuitNet-ISPD20</td>
    <td class="tg-0lax">yes</td>
    <td class="tg-0lax">no</td>
    <td class="tg-0lax">N/A</td>
    <td class="tg-0lax">N/A</td>
  </tr>
</tbody>
</table>


They share a similar directory structure:

```
.               
├── routability_features
| ├── cell_density
| └── congestion
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
| ├── DRC
| | ├── DRC_all
| | └── DRC_seperated
| ├── macro_region
| └── RUDY
|   ├── RUDY
|   ├── RUDY_long
|   ├── RUDY_short
|   ├── RUDY_pin
|   └── RUDY_pin_long
├── IR_drop_features
| ├── power_i
| ├── power_s
| ├── power_sca
| ├── power_all
| ├── power_t
| └── IR_drop
├── graph_features
| ├── graph_information
| └── instance_placement
├── timing_features
| ├── pin_positions
| └── timing_features
├── raw_data
| ├── LEF&DEF
| └── instance_placement
└── script
  ├── decompress_routability.py
  ├── decompress_IR_drop.py
  └── generate_training_set.py
```

  We separate the features and store them in different directories to enable custom applications. Thus they need to be preprocessed and combined in certain arrangement for training.  Our scripts can preprocess and combine different features for training and testing.  But we also encourage to implement different preprocessing methods and use different combinations of features.

  If you meet any problems, feel free to open an issue in our repository or contact us by email.

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