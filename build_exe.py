import PyInstaller.__main__
import os
import shutil

def build():
    print("Building Standalone Executable (.exe) for Stock Market Screener...")
    
    dist_dir = os.path.join(os.path.dirname(__file__), "dist")
    build_dir = os.path.join(os.path.dirname(__file__), "build")

    args = [
        'main.py',
        '--name=StockMarket_AIML_Screener',
        '--onefile',
        '--noconsole',
        '--add-data=config;config',
    ]

    try:
        PyInstaller.__main__.run(args)
        print("\nBuild completed successfully!")
        print(f"Standalone Executable location: {os.path.join(dist_dir, 'StockMarket_AIML_Screener.exe')}")
    except Exception as e:
        print(f"Build failed: {e}")

if __name__ == "__main__":
    build()

