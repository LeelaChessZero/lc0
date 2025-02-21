#!/usr/bin/env python3

import os

# The file path relative to wrap.c location
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "subprojects", "gaviotatb", "compression", "wrap.c")

# Normalize the path to avoid platform-specific issues
file_path = os.path.normpath(file_path)

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: {file_path} does not exist.")
    exit(1)

try:
    # Read the original content
    with open(file_path, 'r') as file:
        content = file.read()

    # Perform the replacement
    modified_content = content.replace('#include "zlib.h"', '#include "zlib/zlib.h"')

    # Check if any changes were made
    if modified_content == content:
        print(f"No changes needed: '#include \"zlib.h\"' not found in {file_path}")
    else:
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(modified_content)
        print(f"Successfully updated {file_path}: Changed '#include \"zlib.h\"' to '#include \"zlib/zlib.h\"'")

except IOError as e:
    print(f"Error processing {file_path}: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(1)