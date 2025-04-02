import csv
import random
import string

with open('gt.txt', 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

# Generate CSV data
rows = []
for idx in range(len(sentences)):
    code, sentence = sentences[idx].split('	')
    rows.append([code, sentence])

# Write to CSV
with open('data/cc100/gt.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['IMAGE_NAME', 'SENTENCE'])
    writer.writerows(rows)

print("CSV file 'spanish_dataset.csv' generated successfully.")
