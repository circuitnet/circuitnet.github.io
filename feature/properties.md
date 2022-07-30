# Overview
## Basic Properties
All features are tile-based. Most information in layout is mapped into tiles with a size of 

1.5$$\mu$$m$$\times$$1.5$$\mu$$m(One exception is the pin confiureation map).
Moreover, layouts are around 450$$\mu$$m$$\times$$450$$\mu$$m, resulting in feature maps of around 300$$\times$$300 tiles. Their detailed calculations are described in the following sections.

The features in dataset are saved seperately and has to be 

Note that the features need to be preprocessed for training, including resizing and normalization. We provide script of our customized preprocessing method used in our experiment, but there is more than one way to complete preprocessing.



## Naming Conventions

10242 samples are generated from 6 original RTL designs with variations in synthesis and physical design as shwon in table below. 

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

The naming convention for data is deined as: {Design name}-{#Macros}-c{Clock}-u{Utilizations}-m{Macro placement}-p{Power mesh setting}-f{filler insertion}

Here is an example of data name: RISCY-a-1-c2-u0.7-m1-p1-f0

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




