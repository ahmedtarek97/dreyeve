# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:46:17 2020

@author: Abdelrahman Al-Wali
"""
import OriginalWindow
from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    window = OriginalWindow.Window()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()