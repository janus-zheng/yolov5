''' autodocstring '''
import argparse

import torch
import cv2


def process_image(weights, input_image):
    """_summary_
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Setting device: {device}")

    yolov5_model = torch.hub.load('/home/lilei/yolov5', 'custom',
                                  path=weights, device=device, source="local")

    img_arr = cv2.imread(input_image)[..., ::-1]  # OpenCV image (BGR to RGB)
    im_h, im_w, _ = img_arr.shape
    results = yolov5_model(img_arr, size=(im_w, im_h))  # inference
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    return 0


def main():
    """ main function """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True,
                        help='The yolov5 model path')
    parser.add_argument('--input-image', type=str, required=True,
                        help='The input image file path')
    args = parser.parse_args()

    process_image(args.weights, args.input_image)


if __name__ == '__main__':
    main()
