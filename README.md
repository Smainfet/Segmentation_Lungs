<h2>
Automatic Segmentation of CT images of lungs affected by COVID-19 with U-Net
</h2>

As part of the course _Medical image processing_, i developed a solution to perform an automatic segmentation of lungs affected by COVID-19.


Due to the scale of people affected by Covid-19 and because the scanners have great inter-variability, I proposed a robust and rapid lung segmentation method, using Deep Learning U-Net architecture, which will serve as a pre-processing step of a **hypothetic** chain of automatic diagnosis, to focus the automatic analysis of the CT scan images on the lung region.

<br />

<div align="center">
<img width="596" alt="Capture d’écran 2023-09-20 à 11 45 41" src="https://github.com/Smainfet/Smainfet/assets/97527246/03de4915-ecc7-4570-8ef0-e0d6697db765">

 

| <img width="400" alt="Capture d’écran 2023-09-20 à 13 09 17" src="https://github.com/Smainfet/Smainfet/assets/97527246/c35d4073-47ae-456e-a22f-d8333788fe90"> | 
|:--:| 
| <sub><sup> *comparison of segmentation results obtained with several methods* </sup></sub>|

</div>

<br />

I used [data available on kaggle](https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans) labeled by expert in segmentation. To improve training, I removed the volumes where there were no lungs.

<br />

<p align="center">
<img width="693" alt="Capture d’écran 2023-09-20 à 02 56 57" class="center" src="https://github.com/Smainfet/Smainfet/assets/97527246/be5e62eb-146c-4543-b2d2-f0c861ae05b6">
</p>


We can imagine that this processing chain will be used to direct health services as quickly and reliably as possible towards the diagnosis of the patient observed, and thus relieve them of this workload.

<br />
<br />
