import datetime
import gc
import logging
import os
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aura.asr.threads import FileTranscriberThread, ModelLoaderThread, TranscriberThread
from aura.audio.denoise import normalize_denoise_preset
from aura.audio.capture import AudioRecorderThread
from aura.audio.export import normalize_wav_to_mp3
from aura.settings import DEFAULT_SETTINGS
from aura.system.update_checker import UpdateCheckerThread
from aura.ui.messages import UI_TEXT

logger = logging.getLogger(__name__)


class TranscriptionTab(QWidget):
    def __init__(self, settings=DEFAULT_SETTINGS, strings=UI_TEXT):
        super().__init__()
        self.settings = settings
        self.strings = strings
        self.recorder_thread = None
        self.file_thread = None
        self.transcriber_thread = TranscriberThread()
        self.transcriber_thread.text_updated.connect(self.update_log)
        self.transcriber_thread.status_updated.connect(self.update_status_only)
        self.transcriber_thread.start()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.pending_files = []
        self.model_loader = None
        self.total_batch_count = 0
        self.update_checker = None

        self.current_folder = os.getcwd()
        self.current_filename = "transcript"
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        info_layout = QHBoxLayout()
        self.status_label = QLabel(self.strings.status_waiting_gpu)
        self.status_label.setStyleSheet("font-weight: bold; color: #00bcd4; font-size: 14px;")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(self.strings.recording_suffix_placeholder)
        info_layout.addWidget(self.status_label, stretch=2)
        info_layout.addWidget(self.name_input, stretch=1)
        layout.addLayout(info_layout)

        self.btn_toggle_settings = QPushButton(self.strings.show_advanced_settings)
        self.btn_toggle_settings.setCheckable(True)
        self.btn_toggle_settings.setStyleSheet(
            "text-align: left; padding: 5px; background: #333; border-radius: 4px; "
            "color: #00bcd4; min-height: 30px;"
        )
        self.btn_toggle_settings.clicked.connect(self.toggle_settings)
        layout.addWidget(self.btn_toggle_settings)

        self.settings_container = QWidget()
        self.settings_container.setVisible(False)
        settings_vbox = QVBoxLayout(self.settings_container)

        denoise_layout = QHBoxLayout()
        self.chk_denoise = QCheckBox(self.strings.denoise_label)
        self.chk_denoise.setChecked(self.settings.denoise_enabled)
        self.chk_denoise.setToolTip(self.strings.denoise_tooltip)
        denoise_layout.addWidget(self.chk_denoise)
        denoise_layout.addStretch()
        settings_vbox.addLayout(denoise_layout)

        norm_layout = QHBoxLayout()
        norm_layout.addWidget(QLabel(self.strings.target_volume_label))
        self.spin_norm = QSpinBox()
        self.spin_norm.setRange(-40, -5)
        self.spin_norm.setValue(int(self.settings.target_dbfs))
        norm_layout.addWidget(self.spin_norm)
        norm_layout.addStretch()
        settings_vbox.addLayout(norm_layout)

        beam_layout = QHBoxLayout()
        beam_layout.addWidget(QLabel(self.strings.beam_size_label))
        self.spin_beam = QSpinBox()
        self.spin_beam.setRange(1, 15)
        self.spin_beam.setValue(self.settings.beam_size)
        beam_layout.addWidget(self.spin_beam)
        beam_layout.addStretch()
        settings_vbox.addLayout(beam_layout)

        prompt_layout = QVBoxLayout()
        prompt_layout.addWidget(QLabel(self.strings.initial_prompt_label))
        self.prompt_input = QLineEdit()
        self.prompt_input.setText(self.settings.file_initial_prompt or "")
        prompt_layout.addWidget(self.prompt_input)
        settings_vbox.addLayout(prompt_layout)

        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel(self.strings.language_label))
        self.combo_lang = QComboBox()
        self.combo_lang.addItem(self.strings.language_auto, None)
        self.combo_lang.addItem(self.strings.language_zh, "zh")
        self.combo_lang.addItem(self.strings.language_en, "en")
        self.combo_lang.addItem(self.strings.language_ja, "ja")
        lang_index = self.combo_lang.findData(self.settings.language)
        self.combo_lang.setCurrentIndex(lang_index if lang_index >= 0 else 0)
        lang_layout.addWidget(self.combo_lang)
        lang_layout.addStretch()
        settings_vbox.addLayout(lang_layout)

        model_settings_layout = QHBoxLayout()
        model_settings_layout.addWidget(QLabel(self.strings.compute_precision_label))
        self.combo_compute = QComboBox()
        self.combo_compute.addItem(self.strings.compute_float16, "float16")
        self.combo_compute.addItem(self.strings.compute_int8, "int8")
        self.combo_compute.addItem(self.strings.compute_float32, "float32")
        compute_index = self.combo_compute.findData(self.settings.compute_type)
        self.combo_compute.setCurrentIndex(compute_index if compute_index >= 0 else 0)
        model_settings_layout.addWidget(self.combo_compute)

        self.btn_reload_model = QPushButton(self.strings.reload_model)
        self.btn_reload_model.setStyleSheet("background-color: #546e7a; color: white;")
        self.btn_reload_model.clicked.connect(self.apply_model_settings)
        model_settings_layout.addWidget(self.btn_reload_model)

        model_settings_layout.addStretch()
        settings_vbox.addLayout(model_settings_layout)
        layout.addWidget(self.settings_container)

        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        layout.addWidget(self.batch_progress)

        self.plot_widget = pg.PlotWidget(title=self.strings.live_waveform_title)
        self.plot_widget.setYRange(-30000, 30000)
        self.plot_data = np.zeros(4000)
        self.curve = self.plot_widget.plot(self.plot_data, pen="c")
        layout.addWidget(self.plot_widget, stretch=1)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setFontPointSize(12)
        layout.addWidget(self.text_area, stretch=2)

        btn_layout = QHBoxLayout()
        self.btn_record = QPushButton(self.strings.start_recording)
        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_record.setFixedHeight(50)
        self.btn_record.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.btn_import = QPushButton(self.strings.import_media)
        self.btn_import.clicked.connect(self.import_file)
        self.btn_import.setFixedHeight(50)

        self.btn_save_txt = QPushButton(self.strings.save_transcript)
        self.btn_save_txt.clicked.connect(self.save_transcript)
        self.btn_save_txt.setFixedHeight(50)

        btn_layout.addWidget(self.btn_record, stretch=2)
        btn_layout.addWidget(self.btn_import, stretch=2)
        btn_layout.addWidget(self.btn_save_txt, stretch=1)
        layout.addLayout(btn_layout)

        self.batch_hint = QLabel(self.strings.batch_hint)
        self.batch_hint.setWordWrap(True)
        self.batch_hint.setStyleSheet("color: #9e9e9e; font-size: 12px;")
        layout.addWidget(self.batch_hint)

        self.apply_model_settings()
        self.check_for_updates()

    def check_for_updates(self):
        self.update_checker = UpdateCheckerThread()
        self.update_checker.found_update.connect(self.show_update_dialog)
        self.update_checker.start()

    def show_update_dialog(self, version, url):
        reply = QMessageBox.question(
            self,
            self.strings.new_version_found,
            self.strings.update_found(version),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            webbrowser.open(url)

    def toggle_settings(self):
        if self.btn_toggle_settings.isChecked():
            self.btn_toggle_settings.setText(self.strings.hide_advanced_settings)
            self.settings_container.setVisible(True)
        else:
            self.btn_toggle_settings.setText(self.strings.show_advanced_settings)
            self.settings_container.setVisible(False)

    def import_file(self):
        if self.transcriber_thread.model is None:
            QMessageBox.warning(self, self.strings.please_wait_title, self.strings.model_not_ready)
            return
        if self.recorder_thread is not None:
            QMessageBox.warning(self, self.strings.error_title, self.strings.stop_recording_before_import)
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            self.strings.select_media_files,
            "",
            self.strings.media_files_filter,
        )
        if files:
            self.pending_files.extend(files)
            self.total_batch_count = len(self.pending_files)
            self.batch_progress.setMaximum(self.total_batch_count)
            self.batch_progress.setValue(0)
            self.batch_progress.setVisible(True)

            self.btn_record.setEnabled(False)
            self.btn_import.setEnabled(False)
            if self.file_thread is None or not self.file_thread.isRunning():
                self.process_next_file()

    def selected_denoise_preset(self) -> str:
        return normalize_denoise_preset(self.chk_denoise.isChecked(), self.settings.denoise_preset)

    def apply_model_settings(self):
        if self.model_loader and self.model_loader.isRunning():
            return

        new_compute = self.combo_compute.currentData()
        self.btn_reload_model.setEnabled(False)
        self.btn_reload_model.setText(self.strings.loading_model)

        self.model_loader = ModelLoaderThread(self.settings.device, new_compute)
        self.model_loader.status_signal.connect(self.update_status_only)
        self.model_loader.error_signal.connect(self.on_model_error)
        self.model_loader.finished_signal.connect(self.on_model_loaded)
        self.model_loader.start()

    @pyqtSlot(object)
    def on_model_loaded(self, new_model):
        if self.transcriber_thread.model:
            del self.transcriber_thread.model
            gc.collect()

        self.transcriber_thread.model = new_model
        active_device = getattr(self.model_loader, "actual_device", self.settings.device)
        active_compute = getattr(self.model_loader, "actual_compute_type", self.combo_compute.currentData())
        self.transcriber_thread.device = active_device
        self.transcriber_thread.compute_type = active_compute

        combo_index = self.combo_compute.findData(active_compute)
        if combo_index >= 0 and combo_index != self.combo_compute.currentIndex():
            self.combo_compute.blockSignals(True)
            self.combo_compute.setCurrentIndex(combo_index)
            self.combo_compute.blockSignals(False)

        self.btn_reload_model.setEnabled(True)
        self.btn_reload_model.setText(self.strings.reload_model)
        self.status_label.setText(self.strings.model_ready(active_device, active_compute))

    @pyqtSlot(str)
    def on_model_error(self, err_msg):
        QMessageBox.critical(self, self.strings.model_loading_failed, err_msg)
        self.btn_reload_model.setEnabled(True)
        self.btn_reload_model.setText(self.strings.reload_model)

    def process_next_file(self):
        if not self.pending_files:
            self.btn_record.setEnabled(True)
            self.btn_import.setEnabled(True)
            self.status_label.setText(self.strings.batch_tasks_completed)
            self.batch_progress.setVisible(False)
            self.total_batch_count = 0
            return

        file_path = self.pending_files.pop(0)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self.current_filename = f"transcript_{base_name}"
        self.current_folder = os.path.dirname(file_path)

        completed = self.total_batch_count - len(self.pending_files)
        self.batch_progress.setValue(completed)

        total_left = len(self.pending_files) + 1
        self.status_label.setText(self.strings.batch_processing(total_left, base_name))

        self.file_thread = FileTranscriberThread(
            self.transcriber_thread.model,
            file_path,
            target_dbfs=float(self.spin_norm.value()),
            beam_size=self.spin_beam.value(),
            initial_prompt=self.prompt_input.text(),
            language=self.combo_lang.currentData(),
            enable_denoise=self.chk_denoise.isChecked(),
            denoise_preset=self.selected_denoise_preset(),
        )
        self.file_thread.text_updated.connect(self.update_log)
        self.file_thread.status_updated.connect(self.update_status_only)
        self.file_thread.error_signal.connect(self.on_file_error)
        self.file_thread.finished_signal.connect(self.process_next_file)
        self.file_thread.start()

    @pyqtSlot(str)
    def on_file_error(self, err_msg):
        QMessageBox.critical(self, self.strings.file_transcription_failed, err_msg)

    def toggle_record(self):
        if self.recorder_thread is None:
            if self.transcriber_thread.model is None:
                QMessageBox.warning(self, self.strings.please_wait_title, self.strings.model_not_ready)
                return

            suffix = self.name_input.text().strip() or "record"
            timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
            base_name = f"{timestamp}_{suffix}"

            self.current_folder = os.path.join(os.getcwd(), base_name)
            os.makedirs(self.current_folder, exist_ok=True)
            self.current_filename = base_name
            full_path = os.path.join(self.current_folder, base_name)

            self.transcriber_thread.update_live_settings(
                beam_size=self.spin_beam.value(),
                language=self.combo_lang.currentData(),
                initial_prompt=self.prompt_input.text(),
            )

            self.recorder_thread = AudioRecorderThread(
                full_path,
                self.transcriber_thread,
                enable_denoise=self.chk_denoise.isChecked(),
                denoise_preset=self.selected_denoise_preset(),
            )
            self.recorder_thread.waveform_signal.connect(self.update_plot)
            self.recorder_thread.finished_signal.connect(self.process_audio)

            self.btn_import.setEnabled(False)
            self.recorder_thread.start()

            self.btn_record.setText(self.strings.stop_recording)
            self.btn_record.setStyleSheet("background-color: #e74c3c; color: white; font-size: 16px; font-weight: bold;")
            self.status_label.setText(self.strings.recording(base_name))
            self.text_area.clear()
        else:
            self.recorder_thread.running = False
            self.recorder_thread.quit()
            self.recorder_thread = None

            self.btn_import.setEnabled(True)
            self.btn_record.setText(self.strings.start_recording)
            self.btn_record.setStyleSheet("font-size: 16px; font-weight: bold;")
            self.status_label.setText(self.strings.recording_finished_processing)

    def save_transcript(self):
        content = self.text_area.toPlainText()
        if not content.strip():
            QMessageBox.warning(self, self.strings.notice_title, self.strings.no_content_to_save)
            return

        default_path = os.path.join(self.current_folder, f"{self.current_filename}.txt")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.strings.save_file,
            default_path,
            self.strings.text_files_filter,
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            QMessageBox.information(self, self.strings.success_title, self.strings.transcript_saved(file_path))

    @pyqtSlot(np.ndarray)
    def update_plot(self, data):
        data_len = len(data)
        plot_len = len(self.plot_data)

        if data_len >= plot_len:
            self.plot_data[:] = data[-plot_len:]
        else:
            self.plot_data = np.roll(self.plot_data, -data_len)
            self.plot_data[-data_len:] = data

        self.curve.setData(self.plot_data)

    @pyqtSlot(str)
    def update_log(self, text):
        self.text_area.append(text)
        self.text_area.verticalScrollBar().setValue(self.text_area.verticalScrollBar().maximum())

    @pyqtSlot(str)
    def update_status_only(self, text):
        self.status_label.setText(text)

    @pyqtSlot(str)
    def process_audio(self, wav_path):
        if "Hardware mounting failed" in wav_path or "No audio recorded" in wav_path:
            return
        self.executor.submit(self._normalization_task, wav_path, float(self.spin_norm.value()))

    def _normalization_task(self, wav_path, target_dbfs):
        try:
            normalize_wav_to_mp3(wav_path, target_dbfs)
        except Exception as e:
            logger.exception("Recording normalization failed: %s", e)

    def stop_threads(self):
        self.transcriber_thread.stop()
        if self.recorder_thread:
            self.recorder_thread.running = False
        if self.file_thread and self.file_thread.isRunning():
            self.file_thread.request_cancel()
            self.file_thread.wait(2000)
