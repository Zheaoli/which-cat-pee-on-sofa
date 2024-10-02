import cv2
import torch
from datetime import timedelta
from ultralytics import YOLO

model = YOLO("yolov5s.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def process_video(
    video_path: str, roi1: tuple[int, int, int, int], roi2: tuple[int, int, int, int]
) -> None:
    fps = get_video_fps(video_path)
    print(f"Video FPS: {fps}")

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % int(fps) != 0:
            continue

        relative_timestamp = timedelta(seconds=frame_count / fps)

        for roi_index, roi in enumerate([roi1, roi2]):
            roi_frame = frame[roi[1] : roi[3], roi[0] : roi[2]]

            results = model(roi_frame)

            cat_detected = False
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 15:
                        cat_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if cat_detected:
                filename = (
                    f"images/cat_detected_roi{roi_index + 1}_{relative_timestamp}.jpg"
                )
                cv2.imwrite(filename, roi_frame)
                print(
                    f"Cat detected in ROI {roi_index + 1} at {relative_timestamp} and saved: {filename}"
                )

                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

                frame[roi[1] : roi[3], roi[0] : roi[2]] = roi_frame

    cap.release()


def main():
    video_path = "demo.mp4"
    roi1 = (563, 321, 1327, 682)
    roi2 = (260, 374, 524, 714)

    process_video(video_path, roi1, roi2)


if __name__ == "__main__":
    main()
