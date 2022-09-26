import cv2 as cv       #open computer vision
import pyautogui      #python GUI controller
import mss() as sct                #multiple screenshots
import numpy as np
import webbrowser
from matplotlib import pyplot as plt
import time

#direct link: "https://www.google.com/fbx?fbx=minesweeper"


def FindStartingArea(difficulty):
    """
    Find the game starting area based on templates using cv and return the bounding box
    """

    if difficulty == "easy":
        print("havnt made a template for easy yet...")
        return

    elif difficulty == "normal":
        minefield_template = cv.imread('./normal_template.png')
        d, w, h = minefield_template.shape[::-1]
        screen = pyautogui.screenshot()
        screen = cv.cvtColor(np.array(screen), cv.COLOR_RGB2BGR)
        matchminefield = cv.matchTemplate(screen,minefield_template,cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matchminefield)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        return top_left,bottom_right

    elif difficulty == "hard":
        minefield_template = cv.imread('./hard_template.png')
        d, w, h = minefield_template.shape[::-1]
        screen = pyautogui.screenshot()
        screen = cv.cvtColor(np.array(screen), cv.COLOR_RGB2BGR)
        matchminefield = cv.matchTemplate(screen,minefield_template,cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matchminefield)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        return top_left,bottom_right

    else:
        print("please select a valid difficulty")

def InitiateMinefield():
    """
    1: open the game in a new tab in the standard webbrowser
    2: find the game area
    3: change the difficulty to hard
    todo:
    add arguments to select different difficulties
    """

    minefield_template = cv.imread('./normal_template.png') # read the minefield template image
    d, w, h = minefield_template.shape[::-1] # get the shape of the template image
    webbrowser.open('https://www.google.com/fbx?fbx=minesweeper',new=0) # open standard webbrowser to google minesweeper
    time.sleep(2.0) # give the page some time to load

    #screen = np.array(mss.mss().grab(mss.mss().monitors[1])) # is mss faster? maybe use later when speed is more important

    screen = pyautogui.screenshot() # take a screenshot
    screen = cv.cvtColor(np.array(screen), cv.COLOR_RGB2BGR) # reverse RGB values to BGR, since cv uses this...
    matchminefield = cv.matchTemplate(screen,minefield_template,cv.TM_CCOEFF) # match the tempalte to the screenshot

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matchminefield) # single out the min and max vaklue locations, max values are matches for chosen method
    top_left = max_loc # top left corner of match location. Is this in pixel number?
    bottom_right = (top_left[0] + w, top_left[1] + h) # bottom right corner of match location

    g_height = bottom_right[1]-top_left[1] # calculate the game height
    g_width = bottom_right[0]-top_left[0] # calculate the game width

    # make a figure to check if the bounding box is found correctly
    #cv.rectangle(screen,top_left, bottom_right, 255, 2)
    #plt.subplot(121),plt.imshow(matchminefield,cmap = 'gray')
    #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(screen,cmap = 'gray')
    #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #plt.show()

    # move the mouse to the pixel location of the difficulty selector
    # do this by moving the mouse to the top left corner and move it a ratio
    # move it 1/10 east and 1/20 south, ratio of total ms-game pixel length

    dif_loc = [0.0,0.0]
    dif_loc[0] = top_left[0]+g_width/10  # define x location of difficulty selector
    dif_loc[1] = top_left[1]+g_height/20 # define y location of difficulty selector

    # you have to divide by 2, since there are multiple pixels per screen coordinate..
    pyautogui.click(dif_loc[0]/2,dif_loc[1]/2) # click the difficulty selector
    pyautogui.moveRel(0,(g_height*0.170)/2)# then move it down a 0.170 ratio of the height to hover over hard
    pyautogui.click() # click on hard

    return top_left, bottom_right

    # initiation finished

def MinesweeperInitiator():
    """ move the cursor to the centre of the field and click to start the game """

    top_left,bottom_right = FindStartingArea(difficulty="hard")


    g_height = bottom_right[1]-top_left[1] #use OOP? you wont have to do it this way...
    g_width = bottom_right[0]-top_left[0]

    centre = [0.0,0.0]
    centre[0] = (bottom_right[0]-g_width/2)/2 # x loc of centre
    centre[1] = (bottom_right[1]-g_height/2)/2 # y loc of centre

    pyautogui.click(centre[0],centre[1])

def MakeMinefieldGrid(difficulty):
    """
    Screenshot the game area and make a grid based on cv detection of grid fields
    """

    #load templates
    #again: use OOP to have global templates instead of loading each time


#def AnalyzeMinefield():

#def ClickMinefield():



#run the game loop
#def playMS():

    # 1: start minesweeper
    # 1a: find the field
    # 2: click random field
    # 2: make the game grid
    # 3: find all possible flags and clicks
    # 4: loop between steps 2-3, check if no valid clicks
    # 5: if no valid clicks: make random choice adjacent to known field?

    # make sure to run timeout timer in loop
