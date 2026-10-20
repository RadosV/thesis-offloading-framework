import cv2
import cv2.aruco as aruco

# Create the dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Generate marker ID 0, 400x400 pixels
marker_image = aruco.generateImageMarker(aruco_dict, 0, 200)

cv2.imwrite("marker_0.png", marker_image)
print("marker_0.png saved.")