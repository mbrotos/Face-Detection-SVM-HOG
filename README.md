![Class](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/class-classified.png "Class")

- [Summary](#summary)
- [Contributors](#contributors)
- [Results](#results)

# Summary

Multi-scale Face Detection using SVM and Histogram of Oriented Gradients features.

# Contributors

- Adam Sorrenti
- Eisa Keramati nejad

# HOG (Histogram of Oriented Gradients)

Consider matrix M a 3x3 image segment representing pixel intensity(I), where (r,c) is a given pixel.
```
        [ 254 143 22 ] 
    M = [ 230 150 25 ]  
        [ 253 154 21 ] 
```

Calculate the gradient of the image: 
$$G_x(r,c)=I(r,c+1)-I(r,c-1)$$ $$G_y(r,c)=I(r-1,c)-I(r+1,c)$$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Intensity_image_with_gradient_images.png/390px-Intensity_image_with_gradient_images.png)

From left to right shows, an intensity image, $G_x$, and $G_y$

## Original Scale
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/class-x1-hog-8x8.png)
## 1/2x Scale
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/class-x0.5-hog-8x8.png)
## 2x Scale
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/class-x2-hog-8x8.png)

# SVM (Support Vector Machine)

# NMS (Non Maximum Suppression)

![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/NMS-mono.png)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/NMS-multi.png)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/NMS-algo.png)


# Results


![Avg Precision](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/average_precision.png "Avg Precision")

![Recall](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/best-recall-falsepos.png "Recall")

