disp("First, we generate a grid of features across the entire image along")
disp("with concatenating the features into 6x6 bins, and classifying them (as if they represent 36x36-pixel faces)");

fprintf('\n')

disp("We increased the number of most condident prediction to improve recall.")
disp("We also verified that it did not exceed the number of predictions found")

fprintf('\n')
disp("After completeing the single window face detection and see bad results with non-max suppression")
disp("we implemented multi-scale face detection on various scale factors.")
fprintf('\n')
disp("Initally, we used inverse powers of 2 as scaling factors and recieved a max")
disp("average percision of 0.340. By reducing the scaling factors increments")
disp("we observed much better average persicion at around 0.520")

fprintf('\n')
disp("Our final optimization was to implement a confidence threshold where we")
disp("ignored any bounding box with a condifence lower than 0.65. This along with")
disp("an overlay threshold of 0.50 resulted in an average percision of 0.584")
fprintf('\n')
disp("Please press enter to view our best precision-recall graph:")
pause;

persicionRecall=imread('best-precision-recall.png');
imshow(persicionRecall)

disp("Please press enter to view our best recall-false positive graph:")
pause;
recallFalsePos=imread('best-recall-falsepos.png');
imshow(recallFalsePos)

fprintf('\n')

disp("Please press enter to view our best class image results:")
pause;
fprintf('\n')

disp("By observing the preditions made on the class image we have come up with")
disp("the following confusion matrix roughly quantifying our results.")

cMatrix = imread("class-confusion-matrix.png");
imshow(cMatrix);

disp("Here you can see a percision of about 0.7 and an accuracy of 0.5")



