import cv2
import torch
from datetime import timedelta
from ultralytics import YOLO
import multiprocessing as mp
import os
import glob

model = YOLO("yolov5s.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_video_info(video_path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def process_video_chunk(
    args: tuple[
        str, int, int, tuple[int, int, int, int], tuple[int, int, int, int], float, str
    ]
) -> list[dict]:
    video_path, start_frame, end_frame, roi1, roi2, fps, output_dir = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    results = []
    for frame_count in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % int(fps) != 0:
            continue

        relative_timestamp = timedelta(seconds=frame_count / fps)

        for roi_index, roi in enumerate([roi1, roi2]):
            roi_frame = frame[roi[1] : roi[3], roi[0] : roi[2]].copy()

            yolo_results = model(roi_frame)

            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 15:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        filename = os.path.join(
                            output_dir,
                            f"cat_detected_roi{roi_index + 1}_{relative_timestamp}.jpg",
                        )
                        cv2.imwrite(filename, roi_frame)
                        results.append(
                            {
                                "timestamp": relative_timestamp,
                                "roi": roi_index + 1,
                                "filename": filename,
                            }
                        )

    cap.release()
    return results


def parallel_process_video(
    video_path: str,
    roi1: tuple[int, int, int, int],
    roi2: tuple[int, int, int, int],
    num_processes: int = 25,
    output_dir: str = "",
) -> None:
    fps, total_frames = get_video_info(video_path)
    print(f"Processing {video_path}")
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")

    chunk_size = total_frames // num_processes
    chunks = [
        (i * chunk_size, min((i + 1) * chunk_size, total_frames))
        for i in range(num_processes)
    ]

    pool = mp.Pool(processes=num_processes)
    chunk_args = [
        (video_path, start, end, roi1, roi2, fps, output_dir) for start, end in chunks
    ]
    results = pool.map(process_video_chunk, chunk_args)
    pool.close()
    pool.join()

    all_results = [item for sublist in results for item in sublist]

    all_results.sort(key=lambda x: x["timestamp"])

    for result in all_results:
        print(
            f"Cat detected in ROI {result['roi']} at {result['timestamp']} and saved: {result['filename']}"
        )


def get_mp4_files(directory: str) -> list[str]:
    return glob.glob(os.path.join(directory, "*.mp4"))


def create_output_directory(video_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("images", base_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    mp.set_start_method('spawn')
    video_directory = "/opt/video/unifi/"
    roi1 = (563, 321, 1327, 682)
    roi2 = (260, 374, 524, 714)

    mp4_files = get_mp4_files(video_directory)

    for video_path in mp4_files:
        output_dir = create_output_directory(video_path)
        parallel_process_video(video_path, roi1, roi2, output_dir=output_dir)


if __name__ == "__main__":
    main()
