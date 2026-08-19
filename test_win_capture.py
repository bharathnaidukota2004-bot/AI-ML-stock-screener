import win32gui
import win32ui
import win32con
from PIL import Image

def capture_hwnd(hwnd):
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    # PrintWindow works even when occluded or on virtual display
    result = win32gui.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3) # PW_RENDERFULLCONTENT = 3

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

if __name__ == "__main__":
    import sys, subprocess, time, os
    proc = subprocess.Popen([sys.executable, "main.py"])
    time.sleep(3.0)
    
    hwnd_found = []
    def enum_cb(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if "AI/ML" in t or "Screener" in t:
                hwnd_found.append(hwnd)
        return True
    win32gui.EnumWindows(enum_cb, None)
    
    if hwnd_found:
        print(f"Testing PrintWindow capture for HWND {hwnd_found[0]}...")
        im = capture_hwnd(hwnd_found[0])
        im.save("test_capture.png")
        print("Successfully captured window to test_capture.png!")
    else:
        print("Window not found")
        
    proc.terminate()
