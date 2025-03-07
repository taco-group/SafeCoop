import os
import xml.etree.ElementTree as ET

# Define the directory containing the XML files
directory = "."
output_file = "town05_short_r_all.xml"

if os.path.exists(output_file):
    print(f"Output file {output_file} already exists. Exiting.")
    exit()

# Initialize the root element
root = ET.Element("routes")

# Track route IDs
route_id = 0

# Iterate over XML files in the directory
for filename in sorted(os.listdir(directory)):  # Sort for consistent ordering
    if filename.endswith(".xml"):
        file_path = os.path.join(directory, filename)

        # Parse the XML file
        tree = ET.parse(file_path)
        route_element = tree.getroot().find("route")

        if route_element is not None:
            # Assign a new unique ID
            route_element.set("id", str(route_id))
            root.append(route_element)
            route_id += 1

# Create the final XML tree
tree = ET.ElementTree(root)

# Write to the output file
with open(output_file, "wb") as f:
    tree.write(f, encoding="utf-8", xml_declaration=True)

print(f"Combined XML saved as {output_file}")