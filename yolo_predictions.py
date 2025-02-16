import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_pred():

    def __init__(self,model_path,data_path,backend=cv2.dnn.DNN_BACKEND_OPENCV,target=cv2.dnn.DNN_TARGET_CPU):

        #Load YAML data   
        with open(data_path,mode="r") as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)

        self.labels = data_yaml["names"]
        self.nc = data_yaml["nc"]
        print(self.labels)

        #Load YOLO model
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.model.setPreferableBackend(backend)
        self.model.setPreferableTarget(target)

    def image_prediction(self, image_path,verbose=True):

        #Load image
        img = cv2.imread(image_path) if isinstance(image_path, str) else image_path
        image = img.copy()
        row, col, d = image.shape 

        #Create square matrix for image
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        INPUT_WH = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH, INPUT_WH), swapRB=True, crop=False)

        #Get predictions
        self.model.setInput(blob)
        preds = self.model.forward()
        preds = np.transpose(preds, (0, 2, 1))


        #For each row in the prediction array we have 4 columns (x,y,w,h) for the bounding box and n columns for n class probabilities
        class_ids = []
        confidences = []
        boxes = []
        rows = preds[0].shape[0]
        image_width, image_height, _ = input_image.shape
        x_factor = image_width / INPUT_WH
        y_factor = image_height / INPUT_WH

        for r in range(rows):
            row = preds[0][r]
            #FIX: CONFIDENCE NOT SHOWING IN ARRAY
            classes_scores = row[4:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > 0.3: #if probability higher than 30%
                confidences.append(classes_scores[class_id])
                class_ids.append(class_id)
                #Create bounding box
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        boxes = np.array(boxes).tolist()
        confidences = np.array(confidences).tolist()

        #Draw box
        if len(boxes) > 0 and len(confidences) > 0:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
            if len(indexes) > 0:  # Ensure we have valid indexes
                indexes = indexes.flatten()
                for i in indexes:
                    x, y, w, h = boxes[i]
                    bb_conf = int(confidences[i] * 100)
                    classes_id = class_ids[i]
                    class_name = self.labels[classes_id]
                    colors = self.generate_colors(classes_id)

                    text = f'{class_name}: {bb_conf}%'
                    if verbose: 
                        print(text)
                    text_y = max(30, y)
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors, 2)
                    cv2.rectangle(image, (x, text_y - 30), (x + w, text_y), colors, -1)
                    cv2.putText(image, text, (x, text_y - 10), cv2.FONT_ITALIC, 0.7, (0, 0, 0), 1)

        return image 

    
    def generate_colors(self,ID):
        np.random.seed(10)
        colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])
    

    def real_time_prediction(self, video_path,verbose=True):
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video has ended or cannot be read.")
                break

            pred_image = self.image_prediction(frame,verbose=verbose)  
            cv2.imshow("prediction", pred_image)  

            if cv2.waitKey(1) & 0xFF == 27:  # Exit when ESC is pressed
                break

        cap.release()
        cv2.destroyAllWindows()
            
        

        

