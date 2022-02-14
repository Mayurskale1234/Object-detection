import cv2

img = cv2.resize(cv2.imread("soccer_practice.jpg",0),(0,0), fx=0.5, fy=0.5)
template = cv2.resize(cv2.imread("ball.png", 0),(0,0), fx=0.5, fy=0.5) ##one more pic for shoe, that is 'shoe.png' this
# will show the shoe present in this img.

h, w = template.shape

#among this methods one method will be selected automatically for detection of the object.

methods = [cv2.TM_CCORR, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF_NORMED,
           cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED]

for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method)
    min_val,max_va,min_loc,max_loc = cv2.minMaxLoc(result)
    if method in methods:
        location = max_loc
    else:
        location = min_loc
    bottom_right = (location[0] + w, location[1] + h)
    draw_rec = cv2.rectangle(img2,location,bottom_right,255, 3)

    cv2.imshow("Ball detection", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()