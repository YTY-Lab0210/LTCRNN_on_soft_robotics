"""Synchronously record one flex ADC stream, camera video, and tape-derived angle."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np

from serial_utils import add_serial_arguments, open_serial, parse_flex_adc


def wrap_to_180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def angle_from_vertical(upper: tuple[int, int], lower: tuple[int, int]) -> float:
    return math.degrees(math.atan2(lower[0] - upper[0], lower[1] - upper[1]))


def find_black_tapes(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: int,
    minimum_area: float,
) -> tuple[
    tuple[tuple[int, int], tuple[int, int]] | None,
    tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None,
    np.ndarray,
]:
    roi_x, roi_y, roi_w, roi_h = roi
    cropped = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, threshold)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maximum_area = roi_w * roi_h * 0.30
    candidates: list[dict[str, object]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not minimum_area <= area <= maximum_area:
            continue
        moment = cv2.moments(contour)
        if moment["m00"] == 0:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        center = (
            int(moment["m10"] / moment["m00"]) + roi_x,
            int(moment["m01"] / moment["m00"]) + roi_y,
        )
        candidates.append(
            {
                "area": area,
                "center": center,
                "box": (x + roi_x, y + roi_y, width, height),
            }
        )

    candidates.sort(key=lambda item: float(item["area"]), reverse=True)
    if len(candidates) < 2:
        return None, None, mask
    selected = sorted(candidates[:2], key=lambda item: item["center"][1])  # type: ignore[index]
    centers = (selected[0]["center"], selected[1]["center"])
    boxes = (selected[0]["box"], selected[1]["box"])
    return centers, boxes, mask  # type: ignore[return-value]


def select_roi(frame: np.ndarray) -> tuple[int, int, int, int]:
    selected = cv2.selectROI(
        "Select flex sensor area",
        frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow("Select flex sensor area")
    roi = tuple(int(value) for value in selected)
    if roi[2] <= 0 or roi[3] <= 0:
        raise RuntimeError("No valid ROI selected")
    return roi  # type: ignore[return-value]


def make_output_dir(base: Path) -> Path:
    output_dir = base / f"flex_angle_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_serial_arguments(parser)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--black-threshold", type=int, default=80)
    parser.add_argument("--minimum-tape-area", type=float, default=150.0)
    parser.add_argument("--smooth-alpha", type=float, default=0.25)
    parser.add_argument("--show-mask", action="store_true")
    args = parser.parse_args()

    output_dir = make_output_dir(args.output_dir)
    video_path = output_dir / "camera_with_angle.mp4"
    sensor_path = output_dir / "sensor_raw.csv"
    frame_path = output_dir / "frame_angle_data.csv"

    connection = open_serial(args.port, args.baud, timeout=0.2)
    camera = cv2.VideoCapture(args.camera_index)
    video_writer: cv2.VideoWriter | None = None
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera index {args.camera_index}")
        for _ in range(20):
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Cannot read a camera frame")

        roi = select_roi(frame)
        height, width = frame.shape[:2]
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.video_fps,
            (width, height),
        )
        if not video_writer.isOpened():
            raise RuntimeError("Cannot create the output video")

        connection.reset_input_buffer()
        connection.timeout = 0
        serial_buffer = ""
        latest_sample: tuple[float, int, int, int] | None = None
        smoothed_angle: float | None = None
        zero_angle: float | None = None
        frame_index = 0
        start = time.perf_counter()

        with (
            sensor_path.open("w", newline="", encoding="utf-8") as sensor_file,
            frame_path.open("w", newline="", encoding="utf-8") as frame_file,
        ):
            sensor_writer = csv.writer(sensor_file)
            frame_writer = csv.writer(frame_file)
            sensor_writer.writerow(["computer_time_s", "sample_index", "arduino_time_us", "adc"])
            frame_writer.writerow(
                [
                    "frame_index",
                    "computer_time_s",
                    "tape_detected",
                    "raw_angle_deg",
                    "smoothed_angle_deg",
                    "calibrated_angle_deg",
                    "sensor_sample_index",
                    "sensor_arduino_time_us",
                    "sensor_adc",
                    "sensor_age_ms",
                ]
            )
            print("Recording. C=zero, R=reselect ROI, Q/Esc=stop")

            while True:
                if connection.in_waiting:
                    serial_buffer += connection.read(connection.in_waiting).decode(
                        "utf-8", errors="ignore"
                    )
                while "\n" in serial_buffer:
                    line, serial_buffer = serial_buffer.split("\n", 1)
                    parsed = parse_flex_adc(line.strip())
                    if parsed is None:
                        continue
                    receive_time = time.perf_counter() - start
                    latest_sample = (receive_time, *parsed)
                    sensor_writer.writerow([f"{receive_time:.6f}", *parsed])

                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Cannot read a camera frame")
                frame_time = time.perf_counter() - start
                display = frame.copy()
                roi_x, roi_y, roi_w, roi_h = roi
                cv2.rectangle(
                    display,
                    (roi_x, roi_y),
                    (roi_x + roi_w, roi_y + roi_h),
                    (255, 255, 0),
                    2,
                )
                centers, boxes, mask = find_black_tapes(
                    frame,
                    roi,
                    args.black_threshold,
                    args.minimum_tape_area,
                )

                raw_angle: float | None = None
                calibrated_angle: float | None = None
                if centers is not None and boxes is not None:
                    upper, lower = centers
                    raw_angle = angle_from_vertical(upper, lower)
                    smoothed_angle = (
                        raw_angle
                        if smoothed_angle is None
                        else args.smooth_alpha * raw_angle
                        + (1.0 - args.smooth_alpha) * smoothed_angle
                    )
                    if zero_angle is not None:
                        calibrated_angle = wrap_to_180(smoothed_angle - zero_angle)
                    for x, y, box_w, box_h in boxes:
                        cv2.rectangle(display, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2)
                    cv2.line(display, upper, lower, (0, 255, 0), 3)
                    angle_text = (
                        f"Angle: {calibrated_angle:+.2f} deg"
                        if calibrated_angle is not None
                        else f"Raw: {smoothed_angle:+.2f} deg - press C"
                    )
                    color = (0, 255, 0)
                else:
                    angle_text = "Cannot find two black tapes"
                    color = (0, 0, 255)

                cv2.putText(display, angle_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                sample_index = arduino_time_us = adc = sensor_age_ms = ""
                if latest_sample is not None:
                    receive_time, sample_index, arduino_time_us, adc = latest_sample
                    sensor_age_ms = f"{(frame_time - receive_time) * 1000:.3f}"
                    cv2.putText(
                        display,
                        f"ADC: {adc}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

                frame_writer.writerow(
                    [
                        frame_index,
                        f"{frame_time:.6f}",
                        int(centers is not None),
                        "" if raw_angle is None else f"{raw_angle:.4f}",
                        "" if smoothed_angle is None else f"{smoothed_angle:.4f}",
                        "" if calibrated_angle is None else f"{calibrated_angle:.4f}",
                        sample_index,
                        arduino_time_us,
                        adc,
                        sensor_age_ms,
                    ]
                )
                video_writer.write(display)
                cv2.imshow("Flex angle + ADC", display)
                if args.show_mask:
                    cv2.imshow("Black tape mask", mask)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("c"), ord("C")) and smoothed_angle is not None:
                    zero_angle = smoothed_angle
                if key in (ord("r"), ord("R")):
                    roi = select_roi(frame)
                    smoothed_angle = None
                    zero_angle = None

                frame_index += 1
                if frame_index % 30 == 0:
                    sensor_file.flush()
                    frame_file.flush()
    finally:
        if video_writer is not None:
            video_writer.release()
        camera.release()
        connection.close()
        cv2.destroyAllWindows()
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

