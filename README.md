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

Consider matrix M, a 3x3 image segment representing pixel intensity(I), where (r,c) is a given pixel.
```
        [ 254 143 22 ] 
    M = [ 230 150 25 ]  
        [ 253 154 21 ] 
```

Calculate the gradient of the image: 
$$G_x(r,c)=I(r,c+1)-I(r,c-1)$$ $$G_y(r,c)=I(r-1,c)-I(r+1,c)$$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Intensity_image_with_gradient_images.png/390px-Intensity_image_with_gradient_images.png)

From left to right shows, an intensity image, $G_x$, and $G_y$


Calculate gradient direction and magnitude:
$$Magnitude(µ)=sqrt{G_x^2+G_y^2}$$
$$Angle(θ)=|tan^{-1}(G_y/G_x)|$$

Compute HOG:

![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/HOG.png)

## Original Scale
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/class-x1-hog-8x8.png)
## 1/2x Scale
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/class-x0.5-hog-8x8.png)
## 2x Scale
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/class-x2-hog-8x8.png)

# SVM (Support Vector Machine)

Faces training set: (36x36 pixel images)

![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_faces/caltech_web_crop_02344.jpg)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_faces/caltech_web_crop_02345.jpg)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_faces/caltech_web_crop_02346.jpg)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_faces/caltech_web_crop_02347.jpg)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_faces/caltech_web_crop_02348.jpg)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_faces/caltech_web_crop_02349.jpg)



Not faces set:

![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_notfaces/2211.jpg)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_notfaces/2212.jpg)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_notfaces/2213.jpg)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_notfaces/2214.jpg)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/cropped_training_images_notfaces/2215.jpg)



<img src="https://lh6.googleusercontent.com/wVQECvLrpASDIYsN4upUV9zALVxytbtytwHxDhHAvI5OQdIrs3zTukkjkbweHBdlFY2DN_AYu186_0mcgprwT4LJ6oAGcGPwReRz0hUidmIID7cNy34SqdWSWjh8CWCeG4qL8bE" width="1000" height="420" style="background-color: white"></img>


<img src="https://www.mdpi.com/sensors/sensors-21-04283/article_deploy/html/images/sensors-21-04283-g002.png" width="570" height="420" style="border:5px solid black"></img>



# NMS (Non Maximum Suppression)

![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/NMS-mono.png)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/NMS-multi.png)
![](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/NMS-algo.png)


# Results


![Avg Precision](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/average_precision.png "Avg Precision")

![Recall](https://github.com/mbrotos/Face-Detection-SVM-HOG/blob/main/code_and_images/best-recall-falsepos.png "Recall")

