import cv2
import numpy as np
import matplotlib.pyplot as plt

#Function of traffic light detection
def traffic_light_detection(img):
  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  lower_red1 = np.array([0, 100, 100])
  upper_red1 = np.array([10, 255, 255])
  lower_red2 = np.array([170, 100, 100])
  upper_red2 = np.array([180, 255, 255])

  mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
  mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
  mask_red = cv2.bitwise_or(mask_red1, mask_red2)

  #Plot steps
  plt.figure(figsize=(15, 10))
  plt.subplot(2, 3, 1)
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.title("Original Image")
  plt.axis('off')
  plt.subplot(2, 3, 2)
  plt.imshow(img_hsv)
  plt.title("HSV Image")
  plt.axis('off')
  plt.subplot(2, 3, 3)
  plt.imshow(mask_red, cmap='gray')
  plt.title("Red Mask")
  plt.axis('off')

  result = img.copy()
  result[mask_red > 0] = [0, 0, 255]
  plt.subplot(2, 3, 4)
  plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
  plt.title("Red Detection")
  plt.axis('off')

  #Count how many red pixels
  red_pixels = cv2.countNonZero(mask_red)

  #True if its red enough
  return red_pixels < 50

#Detect vehicles
def vehicle_detection(image,zone):
  #Cut the image to the most probably car area
    x, y, w, h = zone
    cut_zone = image[y:y+h, x:x+w]

#Estimate minimum car area to avoid little objects
    area_zone = w * h
    car = area_zone * 0.01

    img_gray = cv2.cvtColor(cut_zone, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 30, 100)

#Get the last contour, ignoring hierarchy
    contours,_ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Return true for each contour if is bigger than the minimum car size
    for countour in contours:
      if cv2.contourArea(countour) > car:
        return True
    return False

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(cut_zone, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Zone")
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Gray Image")
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plt.imshow(img_canny, cmap='gray')
    plt.title("Canny Image")
    plt.axis('off')

def fines(image_path, zone):

  #Read image
  img = cv2.imread(image_path)

  if img is None:
    print("Could not read image")
    return
#Use the functions already created and store in variables
  red_traffic_light = traffic_light_detection(img)
  car = vehicle_detection(img, zone)

  if red_traffic_light:
    print("Red Traffic Light Detected")
  else:
    print("No Red Traffic Light Detected")
  if car:
    print("Car Detected")
  else:
    print("No Car Detected")

  if red_traffic_light and car:
    #Write "fine" in the image
    cv2.putText(img, "Fine", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #Safe fine in the folder
    cv2.imwrite("fines/fine.jpg", img)
    print("fine")

  else:
    cv2.putText(img, "No Fine", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print("no fine")

  #Show the final result
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img_rgb)
  plt.axis('off')
  plt.show()

#Initial values
def main():
  #img size admitted (640x480)
  crossing_zone = (200, 250, 300, 150)
  fines("traffic_light4.jpg", crossing_zone)

main()