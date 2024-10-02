import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QLineEdit,
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPoint, QRect


class ROISelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ROI Selector")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        self.select_button = QPushButton("Select Video", self)
        self.select_button.clicked.connect(self.select_video)
        button_layout.addWidget(self.select_button)

        self.coordinates_input = QLineEdit(self)
        self.coordinates_input.setReadOnly(True)
        self.coordinates_input.setPlaceholderText("Coordinates will appear here")
        button_layout.addWidget(self.coordinates_input)

        layout.addLayout(button_layout)

        self.start_point = QPoint()
        self.end_point = QPoint()
        self.drawing = False

        self.original_pixmap = None

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi)"
        )
        if file_name:
            self.load_first_frame(file_name)

    def load_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(
                frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            self.original_pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(self.original_pixmap)
        cap.release()

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.image_label.underMouse()
        ):
            self.start_point = self.image_label.mapFrom(
                self, event.position().toPoint()
            )
            self.end_point = self.start_point
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing and self.image_label.underMouse():
            self.end_point = self.image_label.mapFrom(self, event.position().toPoint())
            self.update_roi()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.end_point = self.image_label.mapFrom(self, event.position().toPoint())
            self.drawing = False
            self.update_roi()
            self.show_coordinates()

    def update_roi(self):
        if self.original_pixmap:
            pixmap = self.original_pixmap.copy()
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine))
            painter.drawRect(QRect(self.start_point, self.end_point))
            painter.end()
            self.image_label.setPixmap(pixmap)

    def show_coordinates(self):
        x1 = min(self.start_point.x(), self.end_point.x())
        y1 = min(self.start_point.y(), self.end_point.y())
        x2 = max(self.start_point.x(), self.end_point.x())
        y2 = max(self.start_point.y(), self.end_point.y())
        coordinates_text = f"[{x1}, {y1}, {x2}, {y2}]"
        self.coordinates_input.setText(coordinates_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    roi_selector = ROISelector()
    roi_selector.show()
    sys.exit(app.exec())
