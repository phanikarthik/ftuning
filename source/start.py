from qa_generator import generate_qa_pairs
from config import INPUT_FILE_NAME, STAGE1_OUTPUT

def main():
   #nltk.download('all')
   generate_qa_pairs(INPUT_FILE_NAME, 3, 12, STAGE1_OUTPUT)


if __name__ == "__main__":
    main()