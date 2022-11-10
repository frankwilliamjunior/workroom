import cv2
from skimage import feature

image = cv2.imread("/root/trt_cuda/traditional/circle_lbp.png")

image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

img_LBP = feature.local_binary_pattern(image_gray,8,1,"ror")

while True:
    cv2.imshow("lbp",img_LBP)
    if cv2.waitKey(1000)& 0xff == ord("q"):
        break

