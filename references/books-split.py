import os

# Open the 'shakespeare.txt' file
file = open('shakespeare.txt', 'r', encoding='utf-8')
lines = file.readlines()


# Helper function to validate characters
def valid(char):
    if ord('A') <= ord(char) <= ord('Z'):
        return True
    if ord('a') <= ord(char) <= ord('z'):
        return True
    if ord('0') <= ord(char) <= ord('9'):
        return True
    if ord(' ') == ord(char):
        return True
    return False


# Clean the lines of text
lines = [x.strip() for x in lines]                # Remove leading and trailing whitespace
lines = [x.replace("\t", " ") for x in lines]     # Replace tabs with spaces
lines = [x.replace("  ", " ") for x in lines]     # Replace double spaces with single spaces

# Validate and filter characters
for i in range(len(lines)):
    lines[i] = list(filter(valid, lines[i]))      # Filter out invalid characters
    lines[i] = "".join(lines[i])                  # Join the valid characters back into a string

# Extract titles (adjust the indices as needed)
titles = lines[9:53]                              # Titles are between line 9 and 53
lines = lines[54:]                                # Remove titles from the rest of the text


# Create 'books' directory if it doesn't exist
if not os.path.exists('books'):
    os.makedirs('books')

# Loop through each title and write chapters to separate text files
for i in range(len(titles)):
    start = lines.index(titles[i])                # Find the start of the chapter based on title
    if i < len(titles) - 1:
        end = lines.index(titles[i + 1])          # Find the start of the next chapter
        chapter = lines[start:end]                # Extract lines for this chapter
    else:
        chapter = lines[start:]                   # Last chapter goes until the end of the document

    # Create a file for each chapter inside the 'books' directory
    with open(f"books/{i+1}. {titles[i]}.txt", "w", encoding='utf-8') as file2:
        for line in chapter:
            file2.write(line + "\n")              # Write each line of the chapter

file.close()
