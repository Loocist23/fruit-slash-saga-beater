import cv2
import dxcam
import time

REGION = (0, 0, 1280, 720)
camera = dxcam.create()

while True:
    img = camera.grab(region=REGION)
    if img is None or img.size == 0:
        print("Image vide, vérifiez REGION ou la configuration de dxcam.")
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imshow("Test dxcam", img_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    time.sleep(0.1)

cv2.destroyAllWindows()
