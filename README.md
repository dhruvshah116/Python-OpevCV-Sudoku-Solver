# Python-OpevCV-Sudoku-Solver

OpenCV Sudoku Solver:

1.Preprocess Image
2.Find Contours
3.Find sudoku
4.Classify Digits
5.Find Solutions
6.Overlay Solution to the original image


All the OpenCV part is basically about  image processing.
The sudokusolver.py will contain the mathematical aspect when we have the actual digits .
So once we know our board and have values in each of the cells , we can actually use the ‘sudoku solver’ to find out the final result and plot it back.

PreProcessing the image : here we will convert it into grayscale , apply some thresholding.

Then we will have our next step , i.e , finding the contours.

Once we have our actual contours , our next step will be to find and locate our actual problem , the sudoku puzzle. Now here we will also apply word perspective so we get a 
perfect square in case an image is tilted.

Now , once we have that , we are going to classify each of those digits , the spaces , we need to know all of it. Even a single mistake here would change the answer completely or
in the worst case we will not have any answer at all. So its very important that we have this step done right.

Then we will go to find the solution and then at the end , we will overlay our solution to the original image.

OpenCV-Python is a library of Python bindings designed to solve computer vision problems. . All the OpenCV array structures are converted to and from Numpy arrays.
Tensorflow is an end-to-end open source machine learning platform for everyone.

# Utilities : utlis.py
#preProcess():
This function is used to implement the first step , which is preprocessing of the loaded image.
Here , first we convert the original image to grayscale , then apply Gaussian Blur ,  typically to reduce image noise and reduce detail.
Then by using cv2.adaptiveThreshold() function,apply adaptive threshold to the blurred image. thresholding is the simplest method of segmenting images.
From a grayscale image, thresholding can be used to create binary images.
Then return the image obtained after thresholding.

The next step is finding Contours.

#biggestContour():
Pass the contours found in the previous step to the above function.
Assuming the biggest Contour is our sudoku puzzle , this function first checks and counts the corner points and returns the largest contour points , that is our sudoku puzzle.

#reorder():
The previous function returns the corner points of the biggest contour but in random order , but this function arranges them in an efficient manner which is going to be easy 
to deal with.

#initializePredictionModel():
This function is used to load our already trained model(text detection) .

#getPrediction():
The arguments in the function are 'boxes' and 'model' , the boxes is collection of all the 81 splitted images , which represnt the 81 cells of our sudoku puzzle, and 
use our model on each of those images to predict the digits and store them in a list , if an image is blank , 0 is added.

#splitBoxes():
split the warped perspective image into 81 different images .

#stackImages():
This function is used to stack the array of images on a single window.

#displayNumbers():
To display Solution on the image passed as the argument.

# sudokumain.py
Run this function.
First import all the required modules and prgrams.
Then load the image using OpenCV.
Then create a blank image for testing and debugging , then pass our original image for preprocessing.
The next step is to find all the contours in the image using cv2.findContours() function and then draw those contours on the image using cv2.drawContours().
Next step is to pass all the contour points to the biggestContour() function and get the corner points of the largest contour and then draw the largest contour 
on the image and locate our sudoku puzzle assuming the largest contour is our sudoku puzzle.
WarpPerspective : The perspectives of the images or videos can be aligned to obtain better insights of the images or videos extracting useful information.
Next step is to get the warp perspective of the image using cv2.warpPerspective() function , where , first we prepare the points and then create the matrix using 
cv2.getPerspectiveTransform() function and then pass these to the warpPerspective function.

The next step is to split the image obtained after war Perspective into 81 sub images as for our sudoku puzzle and then predict the digits in each of these cells and 
display the detected digits and generate our board.
Next we employ our sudokuSolver.py , we pass the board to sudokuSolver.solve() function which uses backtracting to generate solution to the puzzle and returns the solved board.
Then we convert the obtained solved board to the structure that we have been dealing with .
We then display the solved digits.
Then we get the inverse Warp perspective of the image displaying the solved digits to the original dimesnions in order to overlay it on the original image.
Inverse warp perspective is obtained by sending the reverse matric used for obtaining the warp perspective.

Next is displaying the images.








