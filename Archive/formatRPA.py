import re

with open('Thanos_RPA.txt', 'r') as input_file:
    lines = input_file.readlines()

lines = lines[8:]
filtered_lines = [line for line in lines if not line.startswith('#')]

# Define a regular expression to match numbers
number_pattern = re.compile(r'[-+]?\d*\.\d+|\d+')

filtered_lines = []
for line in lines:
    # Extract numerical values from the line using the regular expression
    numbers = number_pattern.findall(line)
    # Join the numbers into a comma-separated string and add it to the filtered lines
    filtered_line = (' '.join(numbers) + '\n')
    if filtered_line.strip():
        filtered_lines.append(filtered_line)

with open('Thanos_Filtered.txt', 'w') as output_file:
    output_file.writelines(filtered_lines)