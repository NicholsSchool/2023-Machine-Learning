from networktables import NetworkTables

# Start NetworkTables server
NetworkTables.startServer()

# Create a table and put some values into it
table = NetworkTables.getTable("SmartDashboard")

while( True ):
    table.putNumber("example_number", 42)
