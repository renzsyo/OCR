import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6 import uic

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("IDscanner.ui", self)

        # Connect button to function
        self.pushButton.clicked.connect(self.go_next)

    def go_next(self):
        if self.uploadOption.isChecked():
            self.Form1.setCurrentIndex(2)  # Page 3
        elif self.cameraOption.isChecked():
            self.Form1.setCurrentIndex(1)  # Page 2
        else:
            print("No option selected")

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())