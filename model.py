import os

DATASET_DIR = r"C:\Users\rishi\Downloads\asl_alphabet_test\asl_alphabet_test"

print(os.listdir(DATASET_DIR))
print(os.listdir(os.path.join(DATASET_DIR, "A")))
