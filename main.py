import time

from networktables import NetworkTables

# Set up the client and connect to the server
NetworkTables.initialize()
NetworkTables.startClient('192.168.0.8')
# Get the table
table = NetworkTables.getTable('pieces')

# Read messages from the server
while True:
    time.sleep( 1 )

    piece = table.getString('piece', 'nothing')
    minX = table.getNumber('minX', 0)
    minY = table.getNumber('minY', 0)
    maxX = table.getNumber('maxX', 0)
    maxY = table.getNumber('maxY', 0)

    print('piece', piece)
    print('xmin:', minX)
    print('ymin:', minY)
    print('xmax:', maxX)
    print('ymax:', maxY)


