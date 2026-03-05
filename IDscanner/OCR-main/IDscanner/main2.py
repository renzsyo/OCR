import cv2
from inference import scan_national_id_front
image ="C:/Users/Renzo/Downloads/3df938ef-5a1f-407c-8d16-1cc62329fb56.jpg"
image = cv2.imread(image)

result = scan_national_id_front(image)
print(result)