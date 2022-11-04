import cv2 as cv                        #open computer vision
import pyautogui                        #python GUI controller
from mss import mss as sct              #multiple screenshots
import numpy as np
import webbrowser
from matplotlib import pyplot as plt
import time

#direct link: "https://www.google.com/fbx?fbx=minesweeper"


def FindStartingArea(difficulty):
    """
    Find the game area based on templates using cv and return the bounding box
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
    """

    minefield_template = cv.imread('./normal_whole_template.png') # read the minefield template image
    d, w, h = minefield_template.shape[::-1] # get the shape of the template image
    webbrowser.open('https://www.google.com/fbx?fbx=minesweeper',new=0) # open standard webbrowser to google minesweeper
    time.sleep(2.0) # give the page some time to load

    #screen = np.array(mss.mss().grab(mss.mss().monitors[1])) # is mss faster? maybe use later when speed is more important

    screen = pyautogui.screenshot() # take a screenshot
    screen = cv.cvtColor(np.array(screen), cv.COLOR_RGB2BGR) # reverse RGB values to BGR, since cv uses this...
    matchminefield = cv.matchTemplate(screen,minefield_template,cv.TM_CCOEFF) # match the template to the screenshot

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matchminefield) # single out the min and max vaklue locations, max values are matches for chosen method
    top_left = max_loc # top left corner of match location. This is in screen coordinates
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

def MinesweeperInitiator():
    """ move the cursor to the centre of the field and click to start the game 
    return the bounding box for the chosen difficulty"""

    top_left,bottom_right = FindStartingArea(difficulty="hard")


    g_height = bottom_right[1]-top_left[1] #use OOP? you wont have to do it this way...
    g_width = bottom_right[0]-top_left[0]

    centre = [0.0,0.0]
    centre[0] = (bottom_right[0]-g_width/2)/2 # x loc of centre
    centre[1] = (bottom_right[1]-g_height/2)/2 # y loc of centre

    pyautogui.click(centre[0],centre[1])

    return top_left,bottom_right


def checkField(top_left_f,bottom_right_f):

    w = bottom_right_f[0]-top_left_f[0]
    h = bottom_right_f[1]-top_left_f[1]
    
    # use pyautogui's screenshot function
    #fieldscreen = pyautogui.screenshot(region=(top_left_f[0],top_left_f[1],w,h))
    
    #use mss instead
    area = {"top": top_left_f[0], "left": top_left_f[1], "width": w, "height": h}
    fieldscreen = np.array(sct().grab(area))


    fieldscreen = cv.cvtColor(np.array(fieldscreen), cv.COLOR_RGB2BGR)
    #fieldscreen = cv.cvtColor(np.array(fieldscreen), cv.COLOR_BGR2RGB)

    # maybe reduce the captured area by a few pixels
    # to avoid getting info from neighbouring fields
    
    # make a figure to test if bounding boxes are actually correct
    #top_left_ft = top_left_f[0],top_left_f[1]
    #bottom_right_ft = bottom_right_f[0],bottom_right_f[1]

    #screen = pyautogui.screenshot()
    #screen = cv.cvtColor(np.array(screen),cv.COLOR_RGB2BGR)
    #screen = cv.cvtColor(np.array(screen),cv.COLOR_BGR2RGB)
    #cv.rectangle(screen,top_left_f,bottom_right_f,255,2)
    #plt.subplot(111),plt.imshow(screen,cmap='gray')
    #plt.show()


    #### make picture with mss instead
    top_left_ft = top_left_f[0],top_left_f[1]
    bottom_right_ft = bottom_right_f[0],bottom_right_f[1]

    screen = np.array(sct().shot())
    screen = cv.cvtColor(np.array(screen),cv.COLOR_RGB2BGR)
    screen = cv.cvtColor(np.array(screen),cv.COLOR_BGR2RGB)
    cv.rectangle(screen,top_left_f,bottom_right_f,255,2)
    plt.subplot(111),plt.imshow(screen,cmap='gray')
    plt.show()


    ######### field color values
    #light_grass_rgb = [86,206,170]
    #dark_grass_rgb = [79,199,162]

    #light_clear_rgb = [151,187,221]
    #dark_clear_rgb = [145,176,206]

    #one_rbg = [190.120,61]
    #two_rbg = [60,128,67]
    #three_rbg = [42,56,192]
    #four_rbg = [147,49,110]
    #five_rbg = [40,141,239]

    #flag_rbg = [17,64,225]

    #remember to use mirrored of these



    fieldval = -3


    if cv.countNonZero(cv.inRange(fieldscreen,(61,120,190),(61,120,190))) > 0:
        fieldval = 1 # field contain 1
    elif cv.countNonZero(cv.inRange(fieldscreen,(67,128,60),(67,128,60))) > 0:
        fieldval = 2 # field contain 2
    elif cv.countNonZero(cv.inRange(fieldscreen,(192,56,42),(192,56,42))) > 0:
        fieldval = 3 # field contain 3
    elif cv.countNonZero(cv.inRange(fieldscreen,(110,49,147),(110,49,147))) > 0:
        fieldval = 4 # field contain 4
    elif cv.countNonZero(cv.inRange(fieldscreen,(239,141,40),(239,141,40))) > 0:
        fieldval = 5 # field contain 5
    elif cv.countNonZero(cv.inRange(fieldscreen,(225,64,17),(225,64,17))) > 0:
        fieldval = -1 # field contain flag
    elif cv.countNonZero(cv.inRange(fieldscreen,(162,199,79),(170,206,86))) > 0:
        fieldval = 0 # field is unknown (grass)
    elif cv.countNonZero(cv.inRange(fieldscreen,(206,176,145),(221,187,151))) > 0:
        fieldval = -2 # cleared square

    return fieldval


def categorizeGrid(XX,YY,MFgrid):
    """ categorize the WHOLE grid
    -2 is cleared empty (tan)
    -1 is flagged (mine)
    0 is unknown (grass)
    1 is field containing '1'
    2 is field containing '2'
    3 is ... """

    #XX.shape[0] will be 21 for hard and [2] is 25
    # 24 column and 20 row fields

    for i in range(MFgrid.shape[0]):
        for j in range(MFgrid.shape[1]):
            # vectorize this loop?
            #top_left_f = [0.0,0.0]
            #bottom_right_f = [0.0,0.0]

            #top_left_f[0] = XX[i,j]
            #top_left_f[1] = YY[i,j]
            #bottom_right_f[0] = XX[i+1,j+1]
            #bottom_right_f[1] = YY[i+1,j+1]

            top_left_f = int(XX[i,j]),int(YY[i,j])
            bottom_right_f = int(XX[i+1,j+1]),int(YY[i+1,j+1])

            MFgrid[i,j] = checkField(top_left_f,bottom_right_f)
            print(MFgrid[i,j])



    return MFgrid

def InitiateMinefieldGrid(difficulty,top_left,bottom_right):
    """
    input: difficulty, top left coordinates, bottom right coordinates
    output: Minefield grid
    """
    #time.sleep(1.0) # let the animation play

    if difficulty == "hard":
        #20x24 grid
        MFgrid = np.zeros([20,24],dtype=np.int8) # 0 is unknown

        x_borders = np.linspace(top_left[0],bottom_right[0],num=25) #25 instances from left to right
        y_borders = np.linspace(top_left[1],bottom_right[1],num=21) #21 instances from top to bottom
        XX,YY = np.meshgrid(x_borders,y_borders)

        return XX, YY, MFgrid


def AnalyzeMinefield(XX,YY,MFgrid):

    for i in range(MFgrid.shape[0]):
        for j in range(MFgrid.shape[1]):



            if MFgrid[i,j] == 0 and MFgrid[i-1]

            try
            # only start finding solutions if you find a green tile with neighbouring known fields
                pass

            else:
                pass




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
