import cv2
import numpy as np
import time
from google.colab.patches import cv2_imshow 
import matplotlib.pyplot as plt

path = (input("ENTER YOUR PATH OF THE IMAGE :-"))
img = cv2.imread(path)
dict_1 = { 1:"Greay", 2:"Binary", 3:"blue", 4:"smoothing the image", 5:"blurring the image", 6:"Detection the edge of image", 7:"saturation", 8:"green", 9:"white", 10:"red", 11:"hue", 12:"cropping the image", 13:"doubling the image", 14:"Transposing the image", 15:"Face detection", 16:"Hiatogram of the image", 17:"Pyramid of the image", 18:"Morphological Transformation", 19:"Edge detection", 20:"Image Gradient", 21:"Hough Line Transform of image", 22:"Hough Circle Transform of image", 23:"Template Matching of image", 24:"Fourier Transform of image", 25:"Image Filtering"}

height=img.shape[0]
width=img.shape[1]
for i in range(1,len(dict_1)+1):
    print(i,'.',dict_1.get(i))
while(True):
    num=input('Enter the number which type of image you want:')
    
    if num=='1':
        img1=cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
        cv2_imshow(img1)
    elif num=='2':
        ret,bw=cv2.threshold(img,127,255,cv2.THRESH_BINARY)#through thresholding we will try to provide the value the value through which we can put below the particular value we assign the value 0 and above it will be white. 
        cv2_imshow(bw)
        img1=bw
    elif num=='3':
        B,G,R=cv2.split(img)
        zeros=np.zeros((height,width),dtype="uint8")
        cv2_imshow(cv2.merge([B,zeros,zeros]))
        img1=cv2.merge([B,zeros,zeros])
    elif num=='4':
        bilateral=cv2.bilateralFilter(img,7,20,20)#9 ,75 and 75 are sigma color value and sigma space value affects cordinates space and color space 
        cv2_imshow(bilateral)
        img1=bilateral
    elif num=='5':
          a = int(input('''There are two types of blurring available :- 1. Gaussian Blurring, 2. Median Blurring.
                     Please select your blurring (1 or 2):- '''))
          if a==1:
              gaussian=cv2.GaussianBlur(img,(7,7),0)
              cv2_imshow(gaussian)
              img1=gaussian
          elif a==2:
              median=cv2.medianBlur(img,5)
              cv2_imshow(median)
              img1=median
    elif num=='6':
        canny=cv2.Canny(img,20,170)#This demand two thresholds from us i.e; 20 and 170 this is like lower and upper value 
        cv2_imshow(canny)
        img1=canny
    elif num=='7':
        img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        cv2_imshow(img_HSV[:,:,1])
        img1=img_HSV[:,:,1]
    elif num=='8':
        B,G,R=cv2.split(img)
        zeros=np.zeros((height,width),dtype="uint8")
        cv2_imshow(cv2.merge([zeros,G,zeros]))
        img1=cv2.merge([zeros,G,zeros])
    elif num=='9':
        B,G,R=cv2.split(img)
        zeros=np.zeros((height,width),dtype="uint8")
        cv2_imshow(cv2.merge([zeros,zeros,R]))
        img1=cv2.merge([zeros,zeros,R])
    elif num=='10':
        img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        cv2_imshow(img_HSV[:,:,2])
        img1=img_HSV[:,:,2]
    elif num=='11':
        img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        cv2_imshow(img_HSV[:,:,0])
        img1=img_HSV[:,:,0]
    elif num=='12':
        height,width=img.shape[:2]
        start_row,start_col=int(height*0.10),int(width*0.10)#starting pixel coordinates (topleft,of cropping rectangles)
        end_row,end_col=int(height*0.86),int(width*0.86)#ending pixel coordinates (bottom right),this can be changed
        cropped=img[start_row:end_row,start_col:end_col]
        cv2_imshow(cropped)
        img1=cropped
    elif num=='13':
        resized=cv2.resize(img,(int(img.shape[1]*1.5),int(img.shape[0]*1.5)))#converting the float value into integer value
        cv2_imshow(resized)
        img1=resized
    elif num=='14':
        rotation_image=cv2.transpose(img)#this will covert the image of horizontal pixel elements into vertical pixel elements as in matrix
        cv2_imshow(rotation_image)
        img1=rotation_image
    elif num=='15':
        face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.06,minNeighbors=6)
        for x,y,w,h in faces:
            img1=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2_imshow(img1)
    elif num=="16":
        a = int(input('''There are two types of histogram in our system:- 1.lines, 2.blocks.
                          Please select (1 or 2):-'''))
        if a==1:
            color = ('b','g', 'r')
            for i,col in enumerate(color):
              hist = cv2.calcHist([img],[i],None,[256],[0,256])
              plt.plot(hist,color=col)
              plt.xlim([0,256])
            plt.show()
            img1=img
        elif a==2:
            plt.hist(img.ravel(), bins=25,edgecolor = "black")
            plt.show()
            img1 = img
    elif num=="17":
        a = int(input('''There are two kinds of Image Pyramids:- 1) Gaussian Pyramid and 2) Laplacian Pyramids.
                         Please select (1 or 2):-'''))
        if a==1:
            b = int(input('''Gaussian pyramids using 1.cv.pyrDown() and 2.cv.pyrUp() functions.
                              Please select (1 or 2):-'''))
            if b==1:
                lower_reso = cv2.pyrDown(img)
                cv2_imshow(lower_reso)
                img1= lower_reso
            elif b==2:
                lower_reso = cv2.pyrUp(img)
                cv2_imshow(lower_reso)
                img1=lower_reso
        elif a==2:
            print("currently data is not availble")
            break
    elif num=="18":
        a = int(input('''We have different morphological operations like 1.Erosion, 2.Dilation, 3.Opening, 4.Closing
                        Please select (1, 2, 3, 4):-'''))
        if a==1:
            kernel = np.ones((5,5),np.uint8)
            erosion = cv2.erode(img,kernel,iterations = 1)
            cv2_imshow(erosion)
            img1= erosion
        elif a==2:
            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(img,kernel,iterations = 1)
            cv2_imshow(dilation)
            img1 = dilation
        elif a==3:
            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            cv2_imshow(opening)
            img1= opening
            
        elif a==4:
            kernel = np.ones((5,5),np.uint8)
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            cv2_imshow(closing)
            img1 = closing
    elif num=="19":
        edges = cv2.Canny(img,100,200)
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        img1 = img
    elif num=="20":
        a =int(input(''' we are provides three types of gradient filters or High-pass filters 1.Sobelx, 2.Laplacian and 3.sobely.
                        Please select (1, 2, 3):- '''))
        if a==1:
            sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
            cv2_imshow(sobelx)
            img1 = sobelx
        elif a==2:
            laplacian = cv2.Laplacian(img,cv2.CV_64F)
            cv2_imshow(laplacian)
            img1 = laplacian
        elif a==3:
            sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
            cv2_imshow(sobely)
            img1 = sobely

    elif num=="21":
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/180,200)
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2_imshow(line)
        img1 = line
    elif num=="22":
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
        cv2_imshow(circles)
        img1 = circles
    elif num == "23":
        img2 = img.copy()
        template1 = input("Enter path of template:-")
        template = cv2.imread(template1,0)
        w, h = template.shape[::-1]
        # All the 6 methods for comparison in a list
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        for meth in methods:
            img = img2.copy()
            method = eval(meth)
            # Apply template Matching
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img,top_left, bottom_right, 255, 2)
            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.show()
            img1 = img
    elif num == "24":
        rows, cols = img.shape
        crow,ccol = rows/2 , cols/2
        # create a mask first, center square is 1, remaining all zeros
        mask = np.zeros((rows,cols,2),np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1
        # apply mask and inverse DFT
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        plt.subplot(121),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()  
        img1=img
    elif num=="25" :
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(img,-1,kernel)
        cv2_imshow(dst)
        img1 = dst
    else:
        print('invalid input')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2_imshow(img)
    save=input('Do you want to save?y/n')
    if save=='y':
        file=input('Enter the image name to be saved')
        cv2.imwrite(file+'.jpg',img1)
    elif save=='n':
        print('its ok')
    else:
        print('invalid input')
    a=input('Do you have to take break?y/n')
    if a=='y':
        break
    elif a=='n':
        print('its ok')
    else:
        print('invalid input')
        pass 
    

    



