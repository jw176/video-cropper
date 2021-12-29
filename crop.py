import cv2
import argparse
import ctypes
from tqdm import tqdm


def crop(image):
    original_image = image.copy()

    ref_point = []

    def select_rect(event, x, y, flags, param):
        nonlocal ref_point, image

        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))

        if ref_point:
            image = original_image.copy()

            if len(ref_point) == 1:
                image = cv2.rectangle(image, ref_point[0], (x, y), (0, 255, 0), 1)
            elif len(ref_point) == 2:
                image = cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 1)

            cv2.imshow("Video", image)

        return ref_point

    cv2.setMouseCallback('Video', select_rect)

    while True:
        cv2.imshow("Video", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = original_image.copy()
            ref_point = []

        elif key == 13:  # ENTER
            break

    # close all open windows
    cv2.destroyAllWindows()

    if len(ref_point) != 2:
        return None

    x = [ref_point[0][0], ref_point[1][0]]
    y = [ref_point[0][1], ref_point[1][1]]
    x.sort()
    y.sort()
    return x + y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input video
    # crop dimensions optional
    # output file name
    # optional frame number to show when cropping

    # get the first frame (or selected frame) of the video
    input_video = '1.mp4'
    output_file = '1-cropped.avi'

    user32 = ctypes.windll.user32
    screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    window_size = round(screen_size[0] * 0.8), round(screen_size[1] * 0.8)

    cap = cv2.VideoCapture(input_video)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not cap.isOpened():
        raise ValueError()

    X_SCALE = WIDTH / window_size[0]
    Y_SCALE = HEIGHT / window_size[1]

    index = 0
    crop_frame_index = 0
    frame_to_crop = None
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if index == crop_frame_index:
                frame_resized = cv2.resize(frame, window_size)
                frame_to_crop = frame_resized
                break

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break

        index += 1

    cap.release()
    cv2.destroyAllWindows()

    cv2.imshow("Video", frame_to_crop)
    dimensions = crop(frame_to_crop)
    if dimensions is None:
        exit()

    for i, val in enumerate(dimensions):
        dimensions[i] *= X_SCALE if i < 2 else Y_SCALE
        dimensions[i] = round(dimensions[i])

    left, right, top, bottom = dimensions
    print(f"{left=}, {right=}, {top=}, {bottom=}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError()

    if output_file.endswith(".avi"):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    elif output_file.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError("Unsupported output video file extension")
    out = cv2.VideoWriter(output_file, fourcc, FPS, ((right - left), (bottom - top)))

    with tqdm(total=FRAME_COUNT, desc="Cropping frames") as progress_bar:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                cropped = frame[top:bottom, left:right]
                out.write(cropped)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            else:
                break
            progress_bar.update()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
