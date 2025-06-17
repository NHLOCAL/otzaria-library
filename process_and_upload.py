# process_and_upload.py

import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, login

# --- הגדרות ---
HF_DATASET_REPO = "NHLOCAL/judaic-texts-corpus" 
# התיקיות שברצונך לעבד
DIRECTORIES_TO_PROCESS = ["אוצריא", "extraBooks/אוצריא"]
# טוקן שיגיע מ-GitHub Secrets
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

def process_books():
    """
    סורק את התיקיות, קורא את קבצי הטקסט ומארגן אותם לרשימה.
    """
    all_books_data = []
    
    for start_dir in DIRECTORIES_TO_PROCESS:
        if not os.path.isdir(start_dir):
            print(f"אזהרה: התיקייה '{start_dir}' לא קיימת. מדלג.")
            continue
            
        # os.walk סורק את כל התיקיות והקבצים באופן רקורסיבי
        for root, _, files in os.walk(start_dir):
            for filename in files:
                # נניח שכל הקבצים הם קבצי טקסט המייצגים ספרים
                if filename.endswith(".txt"): # ניתן לשנות אם הסיומת אחרת
                    file_path = os.path.join(root, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # שם הספר הוא שם הקובץ ללא הסיומת
                        book_name = os.path.splitext(filename)[0]
                        
                        # הנתיב היחסי לתיקיית הבסיס יכול לשמש כקטגוריה
                        category_path = os.path.relpath(root, start_dir)
                        if category_path == ".":
                           category_path = "" # אם הקובץ בתיקיית הבסיס

                        book_data = {
                            "book_name": book_name,
                            "content": content,
                            "category": category_path.replace("\\", "/"), # נרמול לשימוש ב- /
                            "source_path": file_path.replace("\\", "/")
                        }
                        all_books_data.append(book_data)
                        
                    except Exception as e:
                        print(f"שגיאה בקריאת הקובץ {file_path}: {e}")

    return all_books_data


def main():
    """
    פונקציה ראשית: מעבדת, ממירה ומעלה ל-Hugging Face.
    """
    if not HF_TOKEN:
        print("שגיאה: משתנה הסביבה HUGGINGFACE_TOKEN אינו מוגדר.")
        print("ודא שהגדרת אותו ב-GitHub Secrets.")
        return

    print("מתחבר ל-Hugging Face Hub...")
    login(token=HF_TOKEN)

    print("מתחיל עיבוד הספרים...")
    books_data = process_books()

    if not books_data:
        print("לא נמצאו ספרים לעיבוד. יוצא.")
        return

    print(f"נמצאו ועובדו {len(books_data)} ספרים.")

    # יצירת DataFrame של Pandas וממנו Dataset של Hugging Face
    df = pd.DataFrame(books_data)
    hf_dataset = Dataset.from_pandas(df)

    print(f"מעלה את ה-Dataset ל: {HF_DATASET_REPO}...")
    # העלאת ה-Dataset. אם הריפו לא קיים, הוא ייווצר
    hf_dataset.push_to_hub(HF_DATASET_REPO, private=False) # שנה ל-True אם אתה רוצה שהמאגר יהיה פרטי

    print("ההעלאה הסתיימה בהצלחה!")


if __name__ == "__main__":
    main()
