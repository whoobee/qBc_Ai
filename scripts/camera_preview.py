#!/usr/bin/env python3
"""Full-screen Raspberry Pi camera preview."""

from picamera2 import Picamera2
from picamera2.previews.qt import QGlPicamera2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QObject, QEvent
import sys
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

app = QApplication(sys.argv)

picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)

preview = QGlPicamera2(picam2)
preview.setWindowFlags(Qt.FramelessWindowHint)
preview.showFullScreen()

class KeyFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Escape, Qt.Key_Q):
                app.quit()
                return True
        return False

key_filter = KeyFilter()
app.installEventFilter(key_filter)

picam2.start()

sys.exit(app.exec_())
