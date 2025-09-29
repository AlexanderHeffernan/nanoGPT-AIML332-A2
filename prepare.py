from datasets import load_dataset

ds = load_dataset("srivats666/cricket-rules")
with open("data/cricket_rules.txt", "w") as f:
    for item in ds["train"]:
        # Each item is a string like "question,answer"
        line = item["text"] if "text" in item else item
        if isinstance(line, str) and ',' in line:
            question, answer = line.split(',', 1)
            f.write(question.strip() + " " + answer.strip() + "\n")