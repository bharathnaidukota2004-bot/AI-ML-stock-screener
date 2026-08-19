import sys
import os

# Ensure local imports resolve cleanly
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from ui.main_window import StockScreenerApp

def main():
    print("========================================================================")
    print("STARTING: AI/ML Stock Market Screening & Quantitative Analysis System")
    print("========================================================================")
    app = StockScreenerApp()
    app.mainloop()

if __name__ == "__main__":
    main()


