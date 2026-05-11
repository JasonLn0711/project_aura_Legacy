import gc
import os
import time

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMenu,
    QStyle,
    QSystemTrayIcon,
    QTabWidget,
)

from aura.metadata import __author__, __organization__, __version__
from aura.ui.splitter_tab import SplitterTab
from aura.ui.transcription_tab import TranscriptionTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initSystemTray()

    def initUI(self):
        self.setWindowTitle(f"Aura Audio Assistant (Project Aura) | v{__version__}")
        self.resize(1000, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_transcription = TranscriptionTab()
        self.tab_splitter = SplitterTab()

        self.tabs.addTab(self.tab_transcription, "📝 Transcribing")
        self.tabs.addTab(self.tab_splitter, "✂️ Track Splitting")

        build_date = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(__file__)))

        self.sys_status = QLabel("Status: Idle | GPU: Allocating...")
        self.sys_status.setStyleSheet("padding: 5px; color: #00ff00; font-weight: bold; font-size: 11px;")
        self.statusBar().addWidget(self.sys_status, 1)

        footer_text = f"© {build_date[:4]}  {__organization__}  |  v{__version__} ({build_date})  |  {__author__}"
        footer = QLabel(footer_text)
        footer.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer.setStyleSheet("padding: 5px; color: #888888; font-size: 11px;")
        self.statusBar().addPermanentWidget(footer)

    def initSystemTray(self):
        self.tray_icon = QSystemTrayIcon(self)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume)
        self.tray_icon.setIcon(icon)

        tray_menu = QMenu()

        show_action = QAction("Show Main Window", self)
        show_action.triggered.connect(self.show_window)

        quit_action = QAction("Exit Program", self)
        quit_action.triggered.connect(self.quit_app)

        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)

        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        self.tray_icon.show()

    def show_window(self):
        self.show()
        self.activateWindow()

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show_window()

    def quit_app(self):
        self.perform_cleanup()
        QApplication.quit()

    def perform_cleanup(self):
        self.tab_transcription.stop_threads()

        t_thread = self.tab_transcription.transcriber_thread
        if hasattr(t_thread, "model") and t_thread.model is not None:
            del t_thread.model
            t_thread.model = None

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        if os.path.exists("temp_transcript.txt"):
            os.remove("temp_transcript.txt")

    def closeEvent(self, event):
        if self.tray_icon.isVisible():
            self.hide()
            self.tray_icon.showMessage(
                "Comprehensive Audio Assistant",
                "Program minimized to tray. Recording and transcription will continue in the background.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
            event.ignore()
        else:
            self.perform_cleanup()
            super().closeEvent(event)
