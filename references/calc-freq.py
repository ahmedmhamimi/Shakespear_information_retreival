import os

# Ensure the BOOKS directory exists at the same level as the script
books_dir = "./BOOKS"
if not os.path.exists(books_dir):
    os.makedirs(books_dir)


# List and sort book files
filesNames = os.listdir(books_dir)
#print(filesNames)
filesNames = sorted(
    filesNames,
    key=lambda x: int(x.split(".")[0])  # Sort by the numeric part of the filename
)

#remove this line
filesNames = filesNames[4:5]

# Open all book files and read their contents
files = [open(os.path.join(books_dir, x), "r", encoding="utf-8") for x in filesNames]
datas = [x.readlines() for x in files]


def preproc(book):
    # Clean and normalize book text
    book = [x.strip() for x in book]  # Remove leading/trailing whitespace
    book = [x.lower() for x in book]  # Convert to lowercase
    return book


# Preprocess each book's content
datas = list(map(preproc, datas))

# Count word frequencies for each book and print
for fileName, data in zip(filesNames, datas):
    print(fileName, len(data))  # Print the filename and number of lines
    data = " ".join(data).strip().split(" ")  # Join lines into a single string and split into words
    freq = dict([(x, data.count(x)) for x in data])
    for k, v in freq.items():
        print(k, v)

# Close all the files after reading
for file in files:
    file.close()