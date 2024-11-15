# Basic Properties
## CircuitNet-N28
All features under the directories 'routability_features' and 'IR_drop_features' are tile-based. Most information in layout is mapped into tiles with a size of 2.25$$\mu$$m$$\times$$2.25$$\mu$$m.
Moreover, layouts are around 450$$\mu$$m$$\times$$450$$\mu$$m, resulting in feature maps of around 300$$\times$$300 tiles. **In summary, most of the feature maps are 2-dimension numpy array [w, h] unless otherwise indicated.** Their detailed calculations are described in the following sections.

Note that the features need to be preprocessed for training, including resizing and normalization. We provide script of our customized preprocessing method used in our experiment, but there is more than one way to complete preprocessing.



### Naming Conventions

10242 samples are generated for feature extraction from 6 original RTL designs with variations in synthesis and physical design as shown in table below. 

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg {width:100px;height:100px}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Design</th>
    <th class="tg-c3ow" colspan="2">Synthesis Variations</th>
    <th class="tg-c3ow" colspan="4">Physical Design Variations</th>
  </tr>
  <tr>
    <th class="tg-c3ow">#Macros</th>
    <th class="tg-c3ow">Frequency<br>(MHz)</th>
    <th class="tg-c3ow">Utilizations<br>(%)</th>
    <th class="tg-c3ow">#Macro<br>Placement</th>
    <th class="tg-c3ow">#Power Mesh<br>Setting</th>
    <th class="tg-c3ow">Filler Insertion</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">RISCY-a</td>
    <td class="tg-c3ow" rowspan="3"><br><br>3/4/5</td>
    <td class="tg-c3ow" rowspan="6"><br><br><br><br><br>50/200/500<br></td>
    <td class="tg-c3ow" rowspan="6"><br><br><br><br><br>70/75/80/85/90</td>
    <td class="tg-c3ow" rowspan="6"><br><br><br><br><br>3</td>
    <td class="tg-c3ow" rowspan="6"><br><br><br><br><br>8</td>
    <td class="tg-c3ow" rowspan="6"><br><br><br><br>After Placement<br>/After Routing</td>
  </tr>
  <tr>
    <td class="tg-c3ow">RISCY-FPU-a</td>
  </tr>
  <tr>
    <td class="tg-c3ow">zero-riscy-a</td>
  </tr>
  <tr>
    <td class="tg-c3ow">RISCY-b</td>
    <td class="tg-c3ow" rowspan="3"><br><br>13/14/15</td>
  </tr>
  <tr>
    <td class="tg-c3ow">RISCY-FPU-b</td>
  </tr>
  <tr>
    <td class="tg-c3ow">zero-riscy-b</td>
  </tr>
</tbody>
</table>

**The naming convention for extracted feature maps is defined as: {Design name}-{#Macros}-c{Clock}-u{Utilizations}-m{Macro placement}-p{Power mesh setting}-f{filler insertion}**

Here is an example: RISCY-a-1-c2-u0.7-m1-p1-f0

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg {width:200px;height:200px}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="3">Comparison table</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Design name</td>
    <td class="tg-c3ow" colspan="2">6 RTL designs</td>
  </tr>
  <tr>
    <td class="tg-c3ow">#Macros</td>
    <td class="tg-c3ow">3/4/5 or 13/14/15</td>
    <td class="tg-c3ow">1/2/3</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Clock</td>
    <td class="tg-c3ow">Frequency 500/200/50 MHz</td>
    <td class="tg-c3ow">Clock period 2/5/20 ns</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Utilizations</td>
    <td class="tg-c3ow">70/75/80/85/90%</td>
    <td class="tg-c3ow">0.7/0.75/0.8/0.85/0.9</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Macro placement</td>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">1/2/3</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Power mesh setting</td>
    <td class="tg-c3ow">8</td>
    <td class="tg-c3ow">1/2/3/4/5/6/7/8</td>
  </tr>
  <tr>
    <td class="tg-c3ow">filler insertion</td>
    <td class="tg-c3ow">After placement/After routing</td>
    <td class="tg-c3ow">1/0</td>
  </tr>
</tbody>
</table>




## CircuitNet-N14
<!-- All features under directories routability_features and IR_drop_features are tile-based. -->
<!-- Most information in layout is mapped into tiles with a size of 1.5$$\mu$$m$$\times$$1.5$$\mu$$m. -->
<!-- Moreover, layouts are around 450$$\mu$$m$$\times$$450$$\mu$$m, resulting in feature maps of around 300$$\times$$300 tiles. **In summary, most of the feature maps are 2-dimension numpy array [w, h] unless otherwise indicated.** Their detailed calculations are described in the following sections. -->

<!-- Note that the features need to be preprocessed for training, including resizing and normalization.
We provide script of our customized preprocessing method used in our experiment, but there is more than one way to complete preprocessing. -->

### Naming Conventions
10345 samples are generated for feature extraction from 8 original RTL designs with variations in synthesis and physical design as shown in table below.
<!-- With the parameter settings, the number of generated sampled are fewer than that of expected 10848 due to some technique reasons. -->


<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8" rowspan="2">Designs</th>
    <th class="tg-nrix">Synthesis<br>Variations</th>
    <th class="tg-nrix" colspan="5">Physical Design<br>Variations</th>
  </tr>
  <tr>
    <th class="tg-nrix">#Frequency</th>
    <th class="tg-nrix">#Macro<br>Placement</th>
    <th class="tg-nrix">#Utilization</th>
    <th class="tg-nrix">#Aspect Ratio</th>
    <th class="tg-nrix">#Power Mesh<br>Setting</th>
    <th class="tg-nrix">#Filler<br>Insertion</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">zero-riscy</td>
    <td class="tg-nrix" rowspan="3">3 (50/200/500)<br></td>
    <td class="tg-nrix" rowspan="3">4</td>
    <td class="tg-nrix" rowspan="3">6 (50/55/60/65/70/75)</td>
    <td class="tg-nrix" rowspan="3">3 (1.0/1.5/2.0)</td>
    <td class="tg-nrix" rowspan="3">8</td>
    <td class="tg-nrix" rowspan="8">2(Before Routing/<br>After Routing)</td>
  </tr>
  <tr>
    <td class="tg-baqh">RISCY</td>
  </tr>
  <tr>
    <td class="tg-baqh">RISCY-FPU</td>
  </tr>
  <tr>
    <td class="tg-baqh">OpenC910-1</td>
    <td class="tg-nrix" rowspan="5">2 (200/500)</td>
    <td class="tg-nrix" rowspan="5">2</td>
    <td class="tg-nrix" rowspan="5">4 (50/55/60/65)</td>
    <td class="tg-nrix" rowspan="5">1 (1.0)</td>
    <td class="tg-nrix" rowspan="5">3</td>
  </tr>
  <tr>
    <td class="tg-baqh">Vortex-small</td>
  </tr>
  <tr>
    <td class="tg-baqh">Vortex-large</td>
  </tr>
  <tr>
    <td class="tg-baqh">NVDLA-small</td>
  </tr>
  <tr>
    <td class="tg-baqh">NVDLA-large</td>
  </tr>
</tbody>
</table>




**The naming convention for extracted feature maps is defined as: {Design name}\_freq\_{#freq}\_mp\_{Macro palcement}\_fpu\_{Utilizations}\_fpa\_{Aspect Ratio}\_p\_{Power mesh setting}\_fi\_{Filler insertion stage}**

Here is an example: RISCY_freq_50_mp_1_fpu_60_fpa_1.0_p_7_fi_ar


