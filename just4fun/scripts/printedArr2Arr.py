import re


with open('temp.txt', 'r') as f:
    text = f.read()
    result = re.sub(r'(?<=\d) ', ', ', text)
    result = result.replace(']\n', '],')
    result = result.replace(']],', ']]\n')
    result = re.sub(r'[ \t]+', ' ', result)

with open('temp.txt', 'w') as f:
    f.write(result)
