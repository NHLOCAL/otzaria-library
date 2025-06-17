# process_and_upload.py (גרסה משופרת ויעילה)

import os
from datasets import Dataset, Features, Value
from huggingface_hub import HfApi, login

# --- הגדרות ---
HF_DATASET_REPO = "NHLOCAL/judaic-texts-corpus"
DIRECTORIES_TO_PROCESS = ["אוצריא", "extraBooks/אוצריא"]
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

def book_generator():
    """
    Generator שמניב (yields) ספר אחד בכל פעם, במקום לטעון הכל לזיכרון.
    """
    for start_dir in DIRECTORIES_TO_PROCESS:
        if not os.path.isdir(start_dir):
            print(f"אזהרה: התיקייה '{start_dir}' לא קיימת. מדלג.")
            continue
            
        for root, _, files in os.walk(start_dir):
            for filename in files:
                if filename.endswith(".txt"): # או סיומת אחרת רלוונטית
                    file_path = os.path.join(root, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        book_name = os.path.splitext(filename)[0]
                        category_path = os.path.relpath(root, start_dir)
                        if category_path == ".":
                           category_path = ""

                        # "מניבים" את המידע החוצה במקום להוסיף לרשימה
                        yield {
                            "book_name": book_name,
                            "content": content,
                            "category": category_path.replace("\\", "/"),
                            "source_path": file_path.replace("\\", "/")
                        }
                        
                    except Exception as e:
                        print(f"שגיאה בקריאת הקובץ {file_path}: {e}")

def main():
    if not HF_TOKEN:
        print("שגיאה: משתנה הסביבה HUGGINGFACE_TOKEN אינו מוגדר.")
        return

    print("מתחבר ל-Hugging Face Hub...")
    login(token=HF_TOKEN)

    print("יוצר Dataset מה-generator (ביעילות זיכרון)...")

    # הגדרת סכמה ברורה לדאטסט
    features = Features({
        'book_name': Value('string'),
        'content': Value('string'),
        'category': Value('string'),
        'source_path': Value('string')
    })

    # יצירת ה-Dataset ישירות מה-generator
    hf_dataset = Dataset.from_generator(book_generator, features=features)
    
    # בדיקה מהירה אם הדאטסט ריק
    if not hf_dataset or len(hf_dataset) == 0:
        print("לא נוצר דאטסט (אולי לא נמצאו קבצים). יוצא.")
        return

    print(f"נוצר דאטסט המכיל {len(hf_dataset)} ספרים.")
    print(f"מעלה את ה-Dataset ל: {HF_DATASET_REPO}...")
    
    hf_dataset.push_to_hub(HF_DATASET_REPO, private=False)

    print("ההעלאה הסתיימה בהצלחה!")

if __name__ == "__main__":
    main()
