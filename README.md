# Unet
Use DRIVE dataset and Unet to split image
In order to reslove the problem about offcial Unet model is not easy to unsdertand, According to the original model, I made some modifications in that.

[my item introduce](https://blog.csdn.net/chrnhao/article/details/132776522?spm=1001.2014.3001.5501)(In chinese)

Here is my modified model
![image](https://github.com/StChenHaoGitHub/Unet/assets/94610552/9b979ce8-32d0-4d86-ab03-eb84817570eb)
Here is the original model
![image](https://github.com/StChenHaoGitHub/Unet/assets/94610552/59e2fc34-730e-4857-90b0-fc32f61925c8)
My model's merits:
* Input and output figuresize is same,This is in line with our commen situation.
* Canceled meaningless clipping of convolutional layers, Try to use the samoe convolutin strategy as much as possible.
* Clear and simple structure, Sutiable for beginner to read.

There are the results:
![image](https://github.com/StChenHaoGitHub/Unet/assets/94610552/5cc0247a-bd46-4d69-bb6a-9abb4681e493)
![image](https://github.com/StChenHaoGitHub/Unet/assets/94610552/19223093-5350-48d3-a1ae-83b25d59e097)

if any other questions:
My official email: chenhao@smail.sut.edu.cn
