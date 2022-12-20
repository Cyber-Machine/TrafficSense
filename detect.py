import sys
import time
import cv2
import utils
import numpy 

from ml import Classifier
from ml import Movenet

def check_pose():
    estimation_model = 'movenet_thunder'  
    tracker_type = 'bounding_box'  # ['keypoint', 'bounding_box']
    classification_model = 'pose_classifier.tflite'
    label_file = 'pose_labels.txt'
    camera_id = -1
    width = 800
    height = 800

    # Initialize the pose estimator selected.
    if estimation_model in ['movenet_lightning', 'movenet_thunder']:
        pose_detector = Movenet(estimation_model)
        print("MoveNet Lightning/Thunder model selected.")
    else:
        sys.exit("Error: Model not supported.")

    # Variables to calculate FPS.
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera.
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters.
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (255, 0, 0)  # Blue
    font_size = 1
    font_thickness = 3
    max_detection_results = 2
    fps_avg_frame_count = 10

    # Initialize the classification model.
    if classification_model:
        classifier = Classifier(classification_model, label_file)
        detection_results_to_show = min(max_detection_results, len(classifier.pose_class_names))

    # Continuously capture images from the camera.
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Unable to open camera. Please check your camera settings.")
            sys.exit()

        counter += 1
        image = cv2.flip(image, 1)

        if estimation_model == 'movenet_multipose':
            # Run pose estimation using a MultiPose model.
            list_persons = pose_detector.detect(image)
        else:
            # Run pose estimation using a SinglePose model, and wrap the result in an array.
            list_persons = [pose_detector.detect(image)]

        # Draw keypoints and edges on input image.	
        image = utils.visualize(image, list_persons)

        if classification_model:
            # Run pose classification.
            prob_list = classifier.classify_pose(list_persons[0])

            scores = []
            # Show classification results on the image.
            for i in range(detection_results_to_show):
                class_name = prob_list[i].label
        
                probability = round(prob_list[i].score, 2)
                scores.append(probability)
            
            class_name = prob_list[numpy.argmax(scores)].label
            
            cv2.putText(image, class_name, (75,50), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
        # Uncomment for FPS

        # Calculate the FPS.		
        # if counter % fps_avg_frame_count == 0:
        #     end_time = time.time()
        #     fps = fps_avg_frame_count / (end_time - start_time)
        #     start_time = time.time()

        # # Show the FPS.
        # fps_text = 'FPS = ' + str(int(fps))
        # text_location = (left_margin, row_size)
        # cv2.putText(image , fps_text , text_location , cv2.FONT_HERSHEY_DUPLEX , font_size , text_color , font_thickness)
        
        
        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Classification', 670, 600)
        cv2.imshow('Pose Classification', image)
        cv2.moveWindow('Pose Classification', 0, 0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    check_pose()