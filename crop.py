import cv2
import argparse
import ctypes
from tqdm import tqdm


def crop(image, img_scale):
    original_image = image.copy()

    ref_point = []

    def scale(val):
        nonlocal img_scale
        return int(val * img_scale)

    def select_rect(event, x, y, flags, param):
        nonlocal ref_point, image

        image = original_image.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))

        if ref_point:

            if len(ref_point) == 1:
                second_point = (x, y)
            else:
                second_point = ref_point[1]

            image = cv2.rectangle(image, ref_point[0], second_point, (0, 255, 0), 1)

            img_height, img_width, _ = image.shape

            # draw distances to the edge
            x_values = [ref_point[0][0], second_point[0]]
            y_values = [ref_point[0][1], second_point[1]]
            x_values.sort()
            y_values.sort()
            mid_x = sum(x_values) // 2
            mid_y = sum(y_values) // 2
            image = cv2.line(image, (0, mid_y), (x_values[0], mid_y), (255, 255, 255), 1)
            image = cv2.line(image, (img_width, mid_y), (x_values[1], mid_y), (255, 255, 255), 1)
            image = cv2.line(image, (mid_x, 0), (mid_x, y_values[0]), (255, 255, 255), 1)
            image = cv2.line(image, (mid_x, img_height), (mid_x, y_values[1]), (255, 255, 255), 1)

            cv2.putText(image, f"{scale(x_values[0])}px", ((0 + x_values[0]) // 2, mid_y + 20), cv2.FONT_HERSHEY_PLAIN, 0.9,
                        (255, 255, 255))
            cv2.putText(image, f"{scale(img_width - x_values[1])}px", ((x_values[1] + img_width) // 2, mid_y + 20), cv2.FONT_HERSHEY_PLAIN, 0.9,
                        (255, 255, 255))
            cv2.putText(image, f"{scale(y_values[0])}px", (mid_x + 10, (0 + y_values[0]) // 2), cv2.FONT_HERSHEY_PLAIN, 0.9,
                        (255, 255, 255))
            cv2.putText(image, f"{scale(img_height - y_values[1])}px", (mid_x + 10, (img_height + y_values[1]) // 2), cv2.FONT_HERSHEY_PLAIN, 0.9,
                        (255, 255, 255))

            # draw width and height of the rectangle
            rect_width = x_values[1] - x_values[0]
            rect_height = y_values[1] - y_values[0]

        image = cv2.putText(image, f"x:{scale(x)}px", (x - 20, y + 25), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255))
        image = cv2.putText(image, f"y:{scale(y)}px", (x - 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255))

        cv2.imshow("Video", image)

        return ref_point

    cv2.setMouseCallback('Video', select_rect)

    while True:
        cv2.imshow("Video", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = original_image.copy()
            ref_point = []

        if key == ord("q") or key == 27:  # ESC
            ref_point = []
            break

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
    parser.add_argument('input', type=str, help="The input video file")
    parser.add_argument('-o', '--output', type=str, help="The output cropped video file name")
    parser.add_argument('-f', '--frame', default=0, type=int, help="The reference frame to show when cropping")

    args = parser.parse_args()

    input_video = args.input
    if args.output:
        output_file = args.output
    else:
        filename, extension = input_video.split(".")
        output_file = filename + "_cropped." + extension

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
    SCALE = max(X_SCALE, Y_SCALE)

    index = 0
    crop_frame_index = args.frame
    frame_to_crop = None
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if index == crop_frame_index:
                frame_resized = cv2.resize(frame, (int(WIDTH / SCALE), int(HEIGHT / SCALE)))
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
    dimensions = crop(frame_to_crop, SCALE)
    if dimensions is None:
        exit()

    for i, val in enumerate(dimensions):
        dimensions[i] *= SCALE
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

    print(f"Cropped video created: {output_file}")
