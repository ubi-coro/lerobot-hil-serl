import cv2
import av
import numpy as np

print("PyAV version:", av.__version__)
print("Linked library versions:", av.library_versions)

def main():
    # Create a random color image of size 512x512
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Define ROI coordinates (top-left corner and width/height)
    x, y = 100, 100
    w, h = 200, 200

    # Crop the ROI from the image
    roi = img[y:y+h, x:x+w]

    # Display the random image and the ROI
    cv2.imshow("Random Image", img)
    cv2.imshow("ROI Crop", roi)

    # Wait until a key is pressed, then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
