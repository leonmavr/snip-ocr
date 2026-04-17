#!/usr/bin/env python3
"""
Simple screen snip + OCR tool using PyQt6, mss, OpenCV, and pytesseract.

Usage:
  Activate your virtualenv (if any) and run:
    python snip_ocr.py

After running, drag a rectangle on the screen to capture. The recognized text
will be shown in a dialog and copied to the clipboard.

Dependencies: PyQt6, mss, opencv-python-headless, pytesseract, pyperclip, numpy
"""
import sys
import time
from typing import Tuple

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QWidget,
        QDialog,
        QVBoxLayout,
        QTextEdit,
        QPushButton,
        QLabel,
        QHBoxLayout,
    )
    from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QCursor
    from PyQt6.QtCore import Qt, QRect, QPoint
except Exception as e:
    print("PyQt6 is required. Install with: pip install PyQt6")
    raise

import mss
import numpy as np
import cv2
import pytesseract
import pyperclip

# Tesseract config: change if you need different OCR settings
TESSERACT_CONFIG = "--oem 3 --psm 3"


def preprocess_for_ocr(bgr_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # If background is dark (mean intensity low), invert image
    mean_intensity = float(np.mean(gray))
    if mean_intensity < 127:
        gray = 255 - gray
    # Improve local contrast for low-contrast/dark scenes
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # resize small images to help OCR
    h, w = gray.shape
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
    # denoise and blur slightly
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # adaptive threshold works better for uneven/dark backgrounds
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    # morphological closing to join characters and reduce gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return th


class ResultDialog(QDialog):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OCR Result")
        self.setMinimumSize(600, 300)
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(text)
        layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy to clipboard")
        close_btn = QPushButton("Close")
        copy_btn.clicked.connect(self.copy)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(copy_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def copy(self):
        pyperclip.copy(self.text_edit.toPlainText())


class SnipWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snip OCR")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # capture full virtual screen once so later the overlay can
        # reveal original pixels
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            sct_img = sct.grab(monitor)
            arr = np.array(sct_img)  # BGRA
            if arr.shape[2] == 4:
                bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            else:
                bgr = arr
            # save BGR numpy for later OCR cropping
            self.screen_np = bgr
            # convert to RGB for QImage
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            bytes_per_line = 3 * w
            qimg = QImage(
                rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            ).copy()
            self.screen_pixmap = QPixmap.fromImage(qimg)

        self.showFullScreen()
        # use a crosshair cursor for precise selection
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.start_pos = None
        self.current_pos = None
        self.selecting = False

    def paintEvent(self, event):
        painter = QPainter(self)
        # draw captured screen as background
        painter.drawPixmap(0, 0, self.screen_pixmap)
        # dark overlay across whole screen
        overlay_color = QColor(0, 0, 0, 120)
        painter.fillRect(self.rect(), overlay_color)
        if self.start_pos is None:
            return

        # selection rectangle (normalized)
        rect = QRect(self.start_pos, self.current_pos).normalized()
        # reveal the original pixels inside the selection by drawing that pixmap region
        sel_pix = self.screen_pixmap.copy(rect)
        painter.drawPixmap(rect.topLeft(), sel_pix)
        # draw border around selection
        pen = QPen(QColor(0, 180, 255), 2)
        painter.setPen(pen)
        painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.position().toPoint()
            self.current_pos = self.start_pos
            self.selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.selecting:
            self.current_pos = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.selecting:
            self.current_pos = event.position().toPoint()
            self.selecting = False
            self.update()
            bbox = self._rect_to_bbox(self.start_pos, self.current_pos)
            if bbox[2] > 5 and bbox[3] > 5:
                self.handle_capture_and_ocr(bbox)
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def _rect_to_bbox(self, p1: QPoint, p2: QPoint) -> Tuple[int, int, int, int]:
        x1 = int(min(p1.x(), p2.x()))
        y1 = int(min(p1.y(), p2.y()))
        x2 = int(max(p1.x(), p2.x()))
        y2 = int(max(p1.y(), p2.y()))
        return x1, y1, x2 - x1, y2 - y1

    def handle_capture_and_ocr(self, bbox: Tuple[int, int, int, int]):
        left, top, width, height = bbox
        # crop from the previously captured full-screen numpy array
        img = self.screen_np[top : top + height, left : left + width].copy()
        if img.size == 0:
            text = ""
        else:
            proc = preprocess_for_ocr(img)
            try:
                text = pytesseract.image_to_string(proc, config=TESSERACT_CONFIG)
            except Exception:
                text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)

        dlg = ResultDialog(text)
        dlg.exec()


def main():
    app = QApplication(sys.argv)
    w = SnipWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
