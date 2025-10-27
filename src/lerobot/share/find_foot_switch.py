import evdev

# List all devices
print("Look for 'PCsensor FootSwitch Keyboard' in the list below:")
devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
for device in devices:
    print(f"{device.path}: {device.name}")

# Open the one named "PCsensor FootSwitch Keyboard"
device = evdev.InputDevice('/dev/input/event3')

print(f"\nListening for events from {device.name} ({device.path})...")

for event in device.read_loop():
    if event.type == evdev.ecodes.EV_KEY:
        key_event = evdev.categorize(event)
        print(key_event)