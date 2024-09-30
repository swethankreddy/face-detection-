import cv2
import mediapipe as mp
import argparse

def detect_faces(image, detector):
    """Process an image to detect faces and draw bounding boxes."""
    
    img_height, img_width, _ = image.shape

    # Convert image to RGB format for MediaPipe
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection_results = detector.process(img_rgb)

    if detection_results.detections:
        # Loop through detected faces
        for face in detection_results.detections:
            # Extract bounding box data for the face
            face_data = face.location_data.relative_bounding_box

            # Convert relative coordinates to absolute pixel values
            x_start = int(face_data.xmin * img_width)
            y_start = int(face_data.ymin * img_height)
            box_width = int(face_data.width * img_width)
            box_height = int(face_data.height * img_height)

            # Draw rectangle around the face
            image = cv2.rectangle(image, 
                                  (x_start, y_start), 
                                  (x_start + box_width, y_start + box_height), 
                                  (0, 255, 0), 5)
    return image

def main():
    parser = argparse.ArgumentParser(description="Face detection using MediaPipe")
    parser.add_argument("--mode", default="video", choices=["image", "video", "webcam"], 
                        help="Set operation mode (image, video, webcam)")
    parser.add_argument("--file", default='/path/to/video.mp4', 
                        help="Path to the image/video file for detection")
    
    args = parser.parse_args()

    # Initialize MediaPipe face detection module
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    if args.mode == "image":
        # Load and process a single image
        image = cv2.imread(args.file)
        processed_img = detect_faces(image, face_detector)
        cv2.imshow('Detected Faces', processed_img)
        cv2.waitKey(0)

    elif args.mode == "video":
        # Process a video file frame by frame
        video = cv2.VideoCapture(args.file)
        ret, frame = video.read()

        while ret:
            frame_with_detections = detect_faces(frame, face_detector)
            cv2.imshow('Video Frame', frame_with_detections)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            ret, frame = video.read()

        video.release()

    elif args.mode == "webcam":
        # Access webcam feed for real-time face detection
        webcam = cv2.VideoCapture(0)
        ret, frame = webcam.read()

        while ret:
            frame_with_detections = detect_faces(frame, face_detector)
            cv2.imshow('Webcam Feed', frame_with_detections)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            ret, frame = webcam.read()

        webcam.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
