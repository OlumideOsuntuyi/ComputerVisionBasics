import ctypes

import pyautogui
import win32gui

def get_fore_win():
    active_window_handle = win32gui.GetForegroundWindow()
    text = win32gui.GetWindowText(active_window_handle)

    remove_chars = " -_"

    clean_text = ''.join(c for c in text.lower() if c not in remove_chars)
    return clean_text

def handlemouseturning(x, y):
    # Screen size
    screen_width, screen_height = pyautogui.size()

    # Deadzone limits
    deadzone = 1.0

    def map_input_to_position(value, deadzone, min_screen, max_screen):
        if -deadzone < value < deadzone:
            return None  # Deadzone, no movement
        # Normalize from [-2, 2] to [0, 1]
        # First, clamp between [-2, 2]
        value = max(-2, min(2, value))
        # Shift and scale to [0, 1]
        normalized = (value + 2) / 4.0
        # Return mapped screen position
        return int(min_screen + normalized * (max_screen - min_screen))

    new_x = map_input_to_position(x, deadzone, 0, screen_width)
    new_y = map_input_to_position(y, deadzone, 0, screen_height)

    if new_x is not None and new_y is not None:
        pyautogui.moveTo(new_x, new_y)