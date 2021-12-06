import cv2
import pytesseract

img = cv2.imread('license_plate.JPG')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny_edge = cv2.Canny(gray_img, 170, 200)

contours, new = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # finding contours based on edges and listing the all found contours
contours = sorted(contours, key = cv2.contourArea, reverse = True) [:30] # sorting and then choosing the contours which is not less than 30

# Initializing the contour, lp and coordinates
contour_lp = None
lp = None
x = None
y = None
W = None
h = None

for contour in contours:
    parameter = cv2.arcLength(contour, True) # finding the length of contours
    shape = cv2.approxPolyDP(contour, 0.01 * parameter, True) #approxPolyDP(trying to find a rectangle not other shapes), 0.01 tries to increase accuracy(if we give 0.02, not license plates would have found), True tries to find, false doesnt

    if len(shape) == 4: # controls the 4 corner
        contour_lp = shape
        x, y, w, h = cv2.boundingRect(contour)
        lp = gray_img[y: y+h, x: x+w]
        break

lp = cv2.bilateralFilter(lp, 11, 17, 17) # bilateralFilter removes the noise, before sending the Tesseract
(thresh, lp) = cv2.threshold(lp, 150, 180, cv2.THRESH_BINARY) # threshold emphasises the contrast to find easily black text and white background

txt = pytesseract.image_to_string(lp) # text recognition from clear image

img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
img = cv2.putText(img, txt, (x-100, y-50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 6, cv2.LINE_AA) # (x-100, y-70) is the text location

print("License Plate: ", txt)

cv2.imshow("License Plate Detection", img)
cv2.waitKey(0)