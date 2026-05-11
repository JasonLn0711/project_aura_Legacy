import sys

from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from aura.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")

    window = MainWindow()
    window.show()
    return app.exec()
