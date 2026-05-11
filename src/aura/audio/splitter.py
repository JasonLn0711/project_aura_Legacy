from PyQt6.QtCore import QThread, pyqtSignal

from aura.audio.splitter_pipeline import SplitterSettings, split_audio_file


class SmartSplitterThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, file_path, output_dir, target_minutes=40, tolerance_minutes=5):
        super().__init__()
        self.file_path = file_path
        self.output_dir = output_dir
        self.settings = SplitterSettings(target_minutes=target_minutes, tolerance_minutes=tolerance_minutes)

    def run(self):
        try:
            split_audio_file(
                self.file_path,
                self.output_dir,
                self.settings,
                log_callback=self.log_signal.emit,
                progress_callback=self.progress_signal.emit,
            )
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))
