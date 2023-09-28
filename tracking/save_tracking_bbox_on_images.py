import cv2

# Load your image
image = cv2.imread("your_image.jpg")

# Loop through your detections and draw bounding boxes and IDs
for detection in detections:
    bbox = detection['bbox']  # Replace with the actual bounding box coordinates
    tracking_id = detection['tracking_id']  # Replace with the actual tracking ID
    
    # Draw bounding box
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Write tracking ID
    cv2.putText(image, str(tracking_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display or save the image
cv2.imshow("Tracking Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
