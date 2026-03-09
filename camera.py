import cv2
import torch
from ultralytics.engine.results import Boxes


class Camera:
    def __init__(self, side):
        self.side = side

        if side == "right":
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"Camera in the {self.side} eye could not be opened.")
            exit()
        
        try:
            ret, frame = self.cap.read()
        except Exception as e:
            print(f"Error when reading a frame from {self.side} camera: {e}")
    

    def show(self):
        ret, frame = self.cap.read()
        if not ret:
            print(f"Frame could not be read from {self.side} camera.")
            exit()
        cv2.imshow(f"Robot {self.side} eye", frame)

        return frame


    def annotate(self, model, target_class='', filter_hands=False):
        for _ in range(5):
            self.cap.read()                                   # skip one frame for clearing the buffer
        ret, frame = self.cap.read()
        if not ret:
            print("Frame could not be read.")
            exit()

        results = model(frame)

        if filter_hands:
            filtered_boxes = self.filter_hands(model, results[0])
            results[0].boxes = Boxes(filtered_boxes, results[0].orig_shape)

        annotated = results[0].plot()

        h, w = frame.shape[:2]
        cam_cx, cam_cy = w//2, h//2

        # print(f"Camera {self.side} eye: center=({cam_cx},{cam_cy}), size=({w},{h})")

        # draw centroid
        cv2.circle(annotated, (cam_cx, cam_cy), 3, (0, 255, 0), -1)

        target_coord_difs = None

        # add centroids
        for box in results[0].boxes:
            cx, cy, w, h = box.xywh[0]  # cx, cy = center x, center y
            cx, cy = int(cx), int(cy)
            cls = int(box.cls[0])

            # # draw centroid
            # cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            # # write coordinates next to the point
            # cv2.putText(
            #     annotated,
            #     f"({cx},{cy})",
            #     (cx + 5, cy - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 0, 255),
            #     1
            # )

             

            if model.names[cls] == target_class:           # Replace with any target class name
                # cy_lower = cy + h // 4                      # bit lower so that he looks at the point of contact of tablet with the object
                target_coord_difs = (cam_cx - cx, cam_cy - cy)

        cv2.imshow(f"Robot {self.side} eye", annotated)

        return target_coord_difs


    def filter_hands(self, model, result):
        if 'RightHand' not in model.names.values() or 'LeftHand' not in model.names.values():
            return result.boxes
        
        right_id = next((k for k, v in model.names.items() if v == 'RightHand'), None)
        left_id = next((k for k, v in model.names.items() if v == 'LeftHand'), None)
        filtered_result = []
        right_hands, left_hands = [], []
        right_hand, left_hand = None, None

        for box in result.boxes.data:
            box = box.clone()

            if box[5] == right_id:
                right_hands.append(box)
            elif box[5] == left_id:
                left_hands.append(box)
            # else:
            #     filtered_result.append(box)
        
        right_hands = sorted(right_hands, key=lambda x: x[4], reverse=True)  # sort by confidence
        left_hands = sorted(left_hands, key=lambda x: x[4], reverse=True)    # sort by confidence
            
        if len(right_hands) > 1 and len(left_hands) == 0:
            left_hand = right_hands[1]
            left_hand[5] = left_id
            left_hand[4] = -left_hand[4]        # make confidence negative to indicate it's a duplicate
        elif len(left_hands) > 1 and len(right_hands) == 0:
            right_hand = left_hands[1]
            right_hand[5] = right_id
            right_hand[4] = -right_hand[4]      # make confidence negative to indicate it's a duplicate
        
        if len(right_hands) > 0:
            right_hand = right_hands[0]
        if len(left_hands) > 0:
            left_hand = left_hands[0]
        
        if right_hand is not None:
            filtered_result.append(right_hand)
        if left_hand is not None:
            filtered_result.append(left_hand)
        
        if len(filtered_result) == 0:
            return torch.empty((0, 6), device=result.boxes.data.device)
        return torch.stack(filtered_result)
    
    
    def find(self, model, target_class):
        self.cap.grab()                                     # skip one frame for clearing the buffer
        ret, frame = self.cap.read()
        if not ret:
            print("Frame could not be read.")
            exit()
        
        h, w = frame.shape[:2]
        cam_cx, cam_cy = w//2, h//2

        results = model(frame)

        for box in results[0].boxes:
            cx, cy, w, h = map(int, box.xywh[0])  # cx, cy = center x, center y
            cls = int(box.cls[0])

            if model.names[cls] == target_class:           # Replace with any target class name
                x_dif, y_dif = cam_cx - cx, cam_cy - cy

                return (x_dif, y_dif)

        return None
    
    
    def release(self):
        self.cap.release()