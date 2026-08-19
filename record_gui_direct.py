import sys
import os
import time
import ctypes
import cv2
import numpy as np
from PIL import Image, ImageGrab
import win32gui
import win32ui
import win32con

# Ensure local imports resolve cleanly
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from ui.main_window import StockScreenerApp

def capture_hwnd(hwnd):
    try:
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        w = right - left
        h = bot - top
        if w <= 0 or h <= 0:
            return None

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        # Call PrintWindow via user32 DLL
        pw_result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
        if pw_result == 0:
            ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        # Check if captured image is all black / empty
        extrema = im.getextrema()
        if extrema == ((0, 0), (0, 0), (0, 0)):
            try:
                im = ImageGrab.grab(bbox=(max(0, left), max(0, top), right, bot))
            except Exception:
                pass

        return im
    except Exception as e:
        print("Capture HWND error:", e)
        try:
            left, top, right, bot = win32gui.GetWindowRect(hwnd)
            return ImageGrab.grab(bbox=(max(0, left), max(0, top), right, bot))
        except Exception:
            return None

def main():
    print("==========================================================")
    print("Direct GUI Event Loop Capture & Video Generation...")
    print("==========================================================")

    artifacts_dir = r"C:\Users\mail\.gemini\antigravity-ide\brain\67f4ad5a-317c-4b21-a114-d9879689b761"
    os.makedirs(artifacts_dir, exist_ok=True)

    app = StockScreenerApp()
    app.update_idletasks()
    app.update()

    hwnd = app.winfo_id()
    print(f"Tkinter Window HWND: {hwnd}")

    # Bring window to front
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception as e:
        print("Window focus note:", e)

    time.sleep(1.0)

    frames = []
    fps = 4.0
    duration = 18.0 # 18 seconds total
    total_frames = int(duration * fps)

    print(f"Recording {total_frames} live demonstration frames across all 3 tabs...")

    for i in range(total_frames):
        t_elapsed = i / fps

        # Programmatically switch tabs to show all features in action
        if t_elapsed < 6.0:
            app.tabview.set("📈 Live 1-Min Screener")
        elif 6.0 <= t_elapsed < 12.0:
            app.tabview.set("🤖 Day 1 ML Training & Features")
        else:
            app.tabview.set("📊 Live Paper Trading & Win/Loss Stats")

        app.update_idletasks()
        app.update()

        img = capture_hwnd(hwnd)
        if img:
            w, h = img.size
            if w > 1400:
                img = img.resize((1280, int(h * 1280 / w)), Image.Resampling.LANCZOS)
            frames.append(img)

        time.sleep(1.0 / fps)

    app.destroy()
    print(f"Captured {len(frames)} frames. Encoding screen recording outputs...")

    if not frames:
        print("Error: No frames captured.")
        return

    # Save WebP
    webp_path = os.path.join(project_dir, "screen_recording.webp")
    webp_artifact = os.path.join(artifacts_dir, "screen_recording.webp")
    frames[0].save(
        webp_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    frames[0].save(
        webp_artifact,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print("Saved WebP recording:", webp_path)

    # Save GIF
    gif_path = os.path.join(project_dir, "screen_recording.gif")
    gif_artifact = os.path.join(artifacts_dir, "screen_recording.gif")
    gif_frames = [f.quantize(colors=256) for f in frames]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    gif_frames[0].save(
        gif_artifact,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print("Saved GIF recording:", gif_path)

    # Save MP4 video
    try:
        mp4_path = os.path.join(project_dir, "screen_recording.mp4")
        mp4_artifact = os.path.join(artifacts_dir, "screen_recording.mp4")

        fw, fh = frames[0].size
        fw = fw if fw % 2 == 0 else fw - 1
        fh = fh if fh % 2 == 0 else fh - 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_project = cv2.VideoWriter(mp4_path, fourcc, fps, (fw, fh))
        out_artifact = cv2.VideoWriter(mp4_artifact, fourcc, fps, (fw, fh))

        for f in frames:
            f_resized = f.resize((fw, fh), Image.Resampling.LANCZOS)
            cv_img = cv2.cvtColor(np.array(f_resized), cv2.COLOR_RGB2BGR)
            out_project.write(cv_img)
            out_artifact.write(cv_img)

        out_project.release()
        out_artifact.release()
        print("Saved MP4 video recording:", mp4_path)
    except Exception as e:
        print("MP4 encoding note:", e)

    # Save key tab screenshots
    if len(frames) >= 20:
        frames[5].save(os.path.join(artifacts_dir, "screener_tab_demo.png"))
        frames[25].save(os.path.join(artifacts_dir, "ml_training_tab_demo.png"))
        frames[50].save(os.path.join(artifacts_dir, "papertrade_tab_demo.png"))

    print("Screen recording generation complete!")

if __name__ == "__main__":
    main()
