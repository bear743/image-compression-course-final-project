## Overview
This project is the final project of a course. It implements a simple image compression and communication system using wavelet transforms, quantization, entropy coding, and network transmission.

## Figures：
### 1.	R-D curves of the test image:

<img src="Barbara.bmp_rd_curve.png" style='display: block; margin-inline: auto'>

It can be seen that smaller quantization steps require more bitrate, but gets higher PSNR (better image quality).

### 2.	Comparison of original and reconstructed images:

<img src="qs_8.png" style='display: block; margin-inline: auto'>

<center>Quantization step size = 8.0</center>

<br>

<img src="qs_32.png" style='display: block; margin-inline: auto'>

<center>Quantization step size = 32.0</center>

