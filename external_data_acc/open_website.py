import tkinter as tk
import subprocess
import time
import win32gui
import win32con
import win32process
import threading
import psutil



# Dictionary to track running apps by URL
_running_apps = {}
_app_lock = threading.Lock()

def embed_app(urls=None):
    if urls is None:
        urls = ["https://www.google.com"]
    
    global _running_apps
    
    with _app_lock:
        # Check if any of the URLs are already running
        for url in urls:
            if url in _running_apps and _running_apps[url]:
                return
        # Mark all URLs as running
        for url in urls:
            _running_apps[url] = True

    def run_app():
        root = tk.Tk()
        root.title(f"Multiple Websites")
        root.geometry("1080x1920+850+1")
        root.minsize(1080, 1920)  # Optional: Set a minimum window size
        root.overrideredirect(True)
        
        root.wm_attributes('-transparentcolor', 'black')

        # Create a frame to embed the application with a default size
        app_frame = tk.Frame(root, width=1080, height=1920, background="black")
        app_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Check if Chrome is already running
        chrome_pid = None
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] and 'chrome.exe' in proc.info['name'].lower():
                chrome_pid = proc.info['pid']
                break

        # Start Chrome or open new tabs in existing Chrome
        try:
            if chrome_pid:
                # Chrome is running, open URLs in new tabs
                for url in urls:
                    chrome_cmd = ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"]
                    chrome_cmd.extend(["--new-tab", url])
                    proc = subprocess.Popen(chrome_cmd)
            else:
                # Start new Chrome instance
                chrome_cmd = ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"]
                chrome_cmd.extend(["--new-window", "--start-maximized"])
                chrome_cmd.extend(urls)
                proc = subprocess.Popen(chrome_cmd)
        except FileNotFoundError:
            try:
                if chrome_pid:
                    # Try alternate Chrome path for new tabs
                    for url in urls:
                        chrome_cmd = ["C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"]
                        chrome_cmd.extend(["--new-tab", url])
                        proc = subprocess.Popen(chrome_cmd)
                else:
                    # Try alternate Chrome path for new window
                    chrome_cmd = ["C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"]
                    chrome_cmd.extend(["--new-window", "--start-maximized"])
                    chrome_cmd.extend(urls)
                    proc = subprocess.Popen(chrome_cmd)
            except FileNotFoundError:
                print("Chrome not found. Please check if Chrome is installed.")
                root.destroy()
                with _app_lock:
                    for url in urls:
                        _running_apps[url] = False
                return
        
        time.sleep(2)  # Increased wait time for Chrome to initialize

        # Find the window handle of the external application
        def enum_windows_callback(hwnd, pid):
            try:
                if win32process.GetWindowThreadProcessId(hwnd)[1] == pid:
                    # Set the parent of the external window to the app_frame
                    win32gui.SetParent(hwnd, app_frame.winfo_id())
                    
                    # Set window style to make it behave as a child window
                    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                    style = style | win32con.WS_CHILD
                    win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
                    
                    # Resize the embedded window to fit the frame
                    win32gui.MoveWindow(hwnd, 0, 0, 1080, 1920, True)
                    return False  # Stop enumeration
            except Exception as e:
                print(f"Error in callback: {e}")
            return True

        # Give Chrome a moment to create its window before trying to embed it
        time.sleep(1)
        try:
            win32gui.EnumWindows(lambda hwnd, param: enum_windows_callback(hwnd, proc.pid), None)
        except Exception as e:
            print(f"Error enumerating windows: {e}")

        # Update the embedded application size when the frame is resized
        def on_resize(event):
            hwnd = None
            def find_hwnd(hwnd_enum, pid):
                nonlocal hwnd
                try:
                    if win32process.GetWindowThreadProcessId(hwnd_enum)[1] == proc.pid:
                        hwnd = hwnd_enum
                        return False
                except Exception:
                    pass
                return True
            
            try:
                win32gui.EnumWindows(find_hwnd, proc.pid)
                if hwnd:
                    win32gui.MoveWindow(hwnd, 0, 0, 1080, 1920, True)
            except Exception as e:
                print(f"Error in resize: {e}")
                
        app_frame.bind("<Configure>", on_resize)

        def on_closing():
            global _running_apps
            with _app_lock:
                for url in urls:
                    _running_apps[url] = False
            try:
                proc.terminate()  # Properly terminate Chrome
            except Exception:
                pass
            root.destroy()
            
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    # Start the app in a separate thread
    app_thread = threading.Thread(target=run_app, daemon=True)
    app_thread.start()


def main():
    # Open multiple URLs in one Chrome window
    urls = [
        "https://www.youtube.com",
        "https://www.facebook.com",
        "https://www.instagram.com"
    ]
    embed_app(urls)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
