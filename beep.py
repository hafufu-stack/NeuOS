# -*- coding: utf-8 -*-
"""Beep notification script. Spawned in a new window by runner.py."""
import winsound, time
print("=== NeuOS: ALL EXPERIMENTS COMPLETE! ===")
for _ in range(5):
    winsound.Beep(1000, 500)
    time.sleep(0.3)
print("Press Enter to close...")
input()
