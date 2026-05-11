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

from aura.system.runtime_paths import remove_transcript_backup
from aura.ui.messages import UI_TEXT
from aura.ui.splitter_tab import SplitterTab
from aura.ui.transcription_tab import TranscriptionTab


class MainWindow(QMainWindow):
    def __init__(self, strings=UI_TEXT):
        super().__init__()
        self.strings = strings
        self.initUI()
        self.initSystemTray()

    def initUI(self):
        self.setWindowTitle(self.strings.window_title)
        self.resize(1000, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_transcription = TranscriptionTab()
        self.tab_splitter = SplitterTab()

        self.tabs.addTab(self.tab_transcription, self.strings.tab_transcribing)
        self.tabs.addTab(self.tab_splitter, self.strings.tab_splitting)

        build_date = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(__file__)))

        self.sys_status = QLabel(self.strings.status_idle_gpu)
        self.sys_status.setStyleSheet("padding: 5px; color: #00ff00; font-weight: bold; font-size: 11px;")
        self.statusBar().addWidget(self.sys_status, 1)

        footer = QLabel(self.strings.footer(build_date))
        footer.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer.setStyleSheet("padding: 5px; color: #888888; font-size: 11px;")
        self.statusBar().addPermanentWidget(footer)

    def initSystemTray(self):
        self.tray_icon = QSystemTrayIcon(self)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume)
        self.tray_icon.setIcon(icon)

        tray_menu = QMenu()

        show_action = QAction(self.strings.tray_show_main_window, self)
        show_action.triggered.connect(self.show_window)

        quit_action = QAction(self.strings.tray_exit_program, self)
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

        remove_transcript_backup()

    def closeEvent(self, event):
        if self.tray_icon.isVisible():
            self.hide()
            self.tray_icon.showMessage(
                self.strings.tray_message_title,
                self.strings.tray_message_body,
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
            event.ignore()
        else:
            self.perform_cleanup()
            super().closeEvent(event)
