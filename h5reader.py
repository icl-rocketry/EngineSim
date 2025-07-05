import h5py

filename = "h5files/20250704-008-release.h5"  

with h5py.File(filename, 'r') as f:
    # List all groups
    # print("Keys: %s" % f.keys())
    
    # Get the first object name
    a_group_key = list(f.keys())[0]
    
    # Get the object type
    a_group = f[a_group_key]
    
    # Iterate through the group items
    for item in a_group.items():
        print(item)
        
    # Access a specific dataset
    dataset = f['channels']['PT290']['data']  # Replace with actual path
    data = dataset[:]
    print(data)  # Print the data or process it as needed