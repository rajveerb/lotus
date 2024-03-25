import json

with open('mapping_funcs.json', 'r') as file:
    data = json.load(file)

def generate_markdown_table(data):
    markdown_table = "| Transformation         | Function                           | Library       |\n"
    markdown_table += "|------------------------|------------------------------------|---------------|\n"
    
    for op, funcs in data['op_to_func'].items():
        if not funcs:
            markdown_table += f"| {op:<23} | No Functions Captured               |               |\n"
            continue
        for func in funcs:
            function, library = func.split("|")
            markdown_table += f"| {op:<23} | {function:<35} | {library:<13} |\n"
            op = ""
        markdown_table += "|                        |                                    |               |\n"

    return markdown_table

markdown_output = generate_markdown_table(data)

with open('transformation_map.md', 'w') as file:
    file.write(markdown_output)