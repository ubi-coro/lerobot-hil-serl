import hid
import pprint

#Get the list of all devices matching this vendor_id/product_id
device_list = hid.enumerate()

pprint.pprint(device_list)