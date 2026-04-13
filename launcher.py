import sys
import argparse
from PyQt6.QtWidgets import QApplication
from GUI.main_window import MainWindow

def main():
    parser = argparse.ArgumentParser(description="FreqTrade AI Bot Launcher")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange to use (binance, okx, bybit)")
    parser.add_argument("--strategy", type=str, default="TrendMLStrategy", help="Strategy to run")
    parser.add_argument("--gui-only", action="store_true", help="Launch only the GUI without the bot backend")
    
    args = parser.parse_args()

    # NOTE: In the future, we will use Python's subprocess module here to 
    # launch the Freqtrade WSL/backend engine in the background.
    # For now, we are just booting up the GUI to build the visual interface!

    print(f"🚀 Launching FreqTrade AI Bot GUI (Targeting: {args.exchange} | {args.strategy})...")
    
    app = QApplication(sys.argv)
    
    # Initialize and show the main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()