print('Setting Up')
import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '3'  # TO get rid of the tensorflow warnings

from utlis import *
import sudokusolver

pathImage = "1.jpg"
heightImg = 450
widthImg = 450

model = initializePredictionModel()  # Load the CNN Model

# 1. Preparing the Image

img = cv2.imread(pathImage)
img = cv2.resize(img, (heightImg, widthImg))  # resizing the image
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Create a blank image for testing debugging
imgThreshold = preProcess(img)

# 2. Find the Contours

imgContours = img.copy()  # Copy Image for Display Purposes (Will contain all the contours)
imgBigContour = img.copy()  # Copy Image for Display Purposes (Will contain the biggest contour)
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)  # Find all the contours
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)  # Draw all the detected contours

# 3. Find the largest contour and use it as our Sudoku Puzzle
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)  # draw the biggest contour
    pts1 = np.float32(biggest)  # prepare points for WARP
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # prepare points
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    # 4. Split the image and find each digit available (Digit Detection)
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    numbers = getPrediction(boxes, model)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    # cv2.imshow("v", imgDetectedDigits)
    numbers = np.asarray(numbers)
    # print(numbers)
    posArray = np.where(numbers > 0, 0,1)  # This places '1' in the empty places and '0' in the places where we have numbers (FOR OUR BOARD)
    # print(posArray)

    # FIND SOLUTION OF THE BOARD
    board = np.array_split(numbers, 9)

    try:
        sudokusolver.solve(board)
    except:
        pass
    #print(board)
    flatlist=[] #to get the solved values in a single list , the way we have been dealing with
    for sublist in board:
        for item in sublist:
            flatlist.append(item)
    solvedNumbers = flatlist*posArray #to get only the solved values in sudoku puzzle and not the original values(only the values which are new)
    #print(solvedNumbers)
    imgSolvedDigits =  displayNumbers(imgSolvedDigits, solvedNumbers)

    #Overlay Solution

    #(Here First We apply Inverse WARP Perspective to get place the solved numbers on the original image's perspective and then overlay it on the original image))
    pts2=np.float32(biggest) #Prepare points for WARP
    pts1=np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # prepare points
    matrix=cv2.getPerspectiveTransform(pts1,pts2) #GER (Inverse Matrix)
    imgInvWarpColored=img.copy()
    imgInvWarpColored=cv2.warpPerspective(imgSolvedDigits,matrix,(widthImg,heightImg))
    inv_perspective=cv2.addWeighted(imgInvWarpColored,1,img,0.5,1)
    imgDetectedDigits=drawGrid(imgDetectedDigits)
    imgSolvedDigits=drawGrid(imgSolvedDigits)

imgArray=([img,imgThreshold,imgContours,imgBigContour],
          [imgDetectedDigits,imgSolvedDigits,imgInvWarpColored,inv_perspective])
stackedImage=stackImages(imgArray,1)
stackedImage=cv2.resize(stackedImage, (700,450), interpolation = cv2.INTER_AREA)
cv2.imshow("stacked Images",stackedImage)




cv2.waitKey(0)
