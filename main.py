import sys
from PyQt5.QtWidgets import QApplication
from gui import AirbaseGUI

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = AirbaseGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()