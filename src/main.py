from db_utils import loadFromDir, save_np_array, load_np_array, save_array, load_array
from db_validation import validateDB

def main():
    # Toggle between generating or loading the database and labels
    regenerate_db = False

    if regenerate_db:
        db, labels = loadFromDir("data/extracted_images")
        save_np_array(db, "database.npy")
        save_array(labels, "labels.pkl")
    else:
        db = load_np_array("database.npy")
        labels = load_array("labels.pkl")

    # Validate the database
    validateDB(db, labels, "data/extracted_images")

if __name__ == "__main__":
    main()
