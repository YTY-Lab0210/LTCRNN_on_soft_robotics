"""Preview a camera stream and verify that OpenCV can access the device."""

from __future__ import annotations

import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera index {args.camera_index}")

        print("Camera opened. Press Q or Esc to exit.")
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Cannot read a camera frame")
            cv2.imshow("Camera test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

