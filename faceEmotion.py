import cv2
from deepface import DeepFace

# Initialize video feed
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with file path for a video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Analyze frame
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    
    # Draw emotion results
    for face in result:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        emotion = face['dominant_emotion']
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display emotion label
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Show video with emotion
    cv2.imshow("Emotion Analysis", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
