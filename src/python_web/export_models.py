import os, json

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

def save_history(history, path):
    with open(path, "w") as f:
        json.dump(history.history, f)

def main():
    """
    Використання:
    - запускати із ноутбука через %run export_models.py після тренування
    - або імпортувати функції і викликати вручну
    """
    print("Це helper. Краще викликати save() і save_history() прямо в ноутбуці.")

if __name__ == "__main__":
    ensure_dirs()
    main()
