import sys
import os
import time
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageGrab
import win32gui
import win32ui
import win32con

def find_app_hwnd():
    matches = []
    def enum_cb(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "AI/ML" in title or "Stock Screener" in title:
                matches.append((hwnd, title))
        return True
    win32gui.EnumWindows(enum_cb, None)
    return matches[0] if matches else (None, "")

def capture_hwnd_printwindow(hwnd):
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

        # PrintWindow with PW_RENDERFULLCONTENT (3)
        result = win32gui.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
        if result == 0:
            result = win32gui.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

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
        return im
    except Exception as e:
        print("Capture HWND error:", e)
        return None

def record_app():
    print("==========================================================")
    print("Starting Automated High-Fidelity Screen Recording...")
    print("==========================================================")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = r"C:\Users\mail\.gemini\antigravity-ide\brain\67f4ad5a-317c-4b21-a114-d9879689b761"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Launch app
    proc = subprocess.Popen([sys.executable, os.path.join(project_dir, "main.py")], cwd=project_dir)
    print(f"App process launched (PID: {proc.pid})")

    # Wait for GUI initialization
    time.sleep(3.5)

    hwnd, title = find_app_hwnd()
    print(f"Target Window: '{title}' (HWND: {hwnd})")

    if hwnd:
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
        except Exception as e:
            print("Focus note:", e)
        time.sleep(1.0)

    frames = []
    fps = 4.0
    duration = 18.0 # 18 seconds
    frame_delay = 1.0 / fps

    start_t = time.time()
    print("Recording live screening, ML model training, and paper trading...")

    while (time.time() - start_t) < duration:
        t_elapsed = time.time() - start_t

        if hwnd:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                x1, y1, x2, y2 = rect

                # Tab switching simulation
                if 5.5 <= t_elapsed <= 5.8:
                    # Switch to Tab 2: Day 1 ML Training
                    import win32api
                    win32api.SetCursorPos((x1 + 330, y1 + 75))
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                elif 11.5 <= t_elapsed <= 11.8:
                    # Switch to Tab 3: Live Paper Trading
                    import win32api
                    win32api.SetCursorPos((x1 + 600, y1 + 75))
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            except Exception:
                pass

            img = capture_hwnd_printwindow(hwnd)
            if img is None:
                try:
                    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                except Exception:
                    img = ImageGrab.grab()
        else:
            img = ImageGrab.grab()

        if img:
            w, h = img.size
            if w > 1400:
                img = img.resize((1280, int(h * 1280 / w)), Image.Resampling.LANCZOS)
            frames.append(img)

        time.sleep(frame_delay)

    # Terminate recorded app
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except Exception:
        proc.kill()

    print(f"Captured {len(frames)} frames. Generating screen recording output files...")

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

    # Save MP4 video using OpenCV
    try:
        mp4_path = os.path.join(project_dir, "screen_recording.mp4")
        mp4_artifact = os.path.join(artifacts_dir, "screen_recording.mp4")

        fw, fh = frames[0].size
        # Make width/height even numbers for H.264
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

    # Save tab screenshots into artifacts
    if len(frames) >= 12:
        frames[3].save(os.path.join(artifacts_dir, "screener_tab_demo.png"))
        frames[8].save(os.path.join(artifacts_dir, "ml_training_tab_demo.png"))
        frames[13].save(os.path.join(artifacts_dir, "papertrade_tab_demo.png"))

    print("Screen recording generation complete!")

if __name__ == "__main__":
    record_app()
