from easyhid import Enumeration

hid = Enumeration()
for device in hid.find():
    print("Device:", device.product_string)
    print("Path:", device.path)
    print("=" * 5)
