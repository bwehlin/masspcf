import numpy as np

def print_np_array_details(name, arr):
    inter = arr.__array_interface__
    # data is a tuple: (pointer_address, read_only_flag)
    address = inter['data'][0] 
    
    print(f"--- {name} ---")
    print(f"Shape: {arr.shape}")
    print(f"Strides: {np.array(arr.strides) / arr.itemsize} items")
    
    if arr.base is not None:
        base_address = arr.base.__array_interface__['data'][0]
        offset = address - base_address
        print(f"Offset from base: {offset / arr.itemsize} items")
    else:
        print("Offset from base: 0 (This is the owner)")
    print("-" * 20)
