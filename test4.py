import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

def get_cursor_pos():
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

ok = user32.SetCursorPos(100, 100)
after = get_cursor_pos()
print("SetCursorPos ok=", ok, "after=", after, flush=True)
