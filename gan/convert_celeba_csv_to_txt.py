
import os
import pandas as pd

# Paths (Assuming script is run from 'gan/' and data is in 'gan/data/celeba/')
DATA_DIR = "./data/celeba"
ATTR_CSV = os.path.join(DATA_DIR, "list_attr_celeba.csv")
EVAL_CSV = os.path.join(DATA_DIR, "list_eval_partition.csv")

ATTR_TXT = os.path.join(DATA_DIR, "list_attr_celeba.txt")
EVAL_TXT = os.path.join(DATA_DIR, "list_eval_partition.txt")

def convert_attr():
    if not os.path.exists(ATTR_CSV):
        print(f"Error: {ATTR_CSV} not found.")
        return

    print(f"Reading {ATTR_CSV}...")
    df = pd.read_csv(ATTR_CSV)
    
    # Kaggle CSV usually has 'image_id' as the first column, then attributes
    # Double check if image_id is index or column
    if 'image_id' in df.columns:
        filenames = df['image_id']
        attrs = df.drop(columns=['image_id'])
    else:
        # Assuming first column is filename
        filenames = df.iloc[:, 0]
        attrs = df.iloc[:, 1:]

    num_imgs = len(df)
    attr_names = " ".join(attrs.columns)
    
    print(f"Writing {ATTR_TXT}...")
    with open(ATTR_TXT, "w") as f:
        # Line 1: Number of images
        f.write(f"{num_imgs}\n")
        # Line 2: Attribute names
        f.write(f"{attr_names}\n")
        
        # Line 3+: Filename + values
        # Iterate and format. Using pandas to_string or similar might be cleaner but 
        # standard loop ensures format controls.
        # This might be slow for 200k images, but safe.
        for idx, row in df.iterrows():
            img_name = row['image_id'] if 'image_id' in df.columns else row.iloc[0]
            # Attributes: Torchvision expects 1 and -1. 
            # Kaggle CSV usually has -1 and 1 already.
            vals = []
            # We iterate columns of 'attrs' dataframe
            for mk in attrs.columns:
                vals.append(str(row[mk]))
            
            line_str = f"{img_name} {' '.join(vals)}\n"
            f.write(line_str)
            
            if idx % 10000 == 0:
                print(f"Processed {idx}/{num_imgs}")

    print("Attribute conversion done.")

def convert_eval():
    if not os.path.exists(EVAL_CSV):
        print(f"Error: {EVAL_CSV} not found.")
        return

    print(f"Reading {EVAL_CSV}...")
    df = pd.read_csv(EVAL_CSV)
    
    print(f"Writing {EVAL_TXT}...")
    with open(EVAL_TXT, "w") as f:
        # No header lines for eval partition in original format?
        # Actually, original file just lists "filename partition_id"
        # No count header.
        
        for idx, row in df.iterrows():
            img_name = row['image_id'] if 'image_id' in df.columns else row.iloc[0]
            partition = row['partition'] if 'partition' in df.columns else row.iloc[1]
            
            f.write(f"{img_name} {partition}\n")

    print("Eval partition conversion done.")

def create_dummy_files(num_imgs):
    # Torchvision CelebA loader requires these files to exist, 
    # even if we don't use them. We create dummy versions.
    
    # 1. identity_CelebA.txt (filename label)
    # 2. list_bbox_celeba.txt (filename x_1 y_1 width height)
    # 3. list_landmarks_align_celeba.txt (filename leye_x leye_y reye_x reye_y nose_x nose_y lmouth_x lmouth_y rmouth_x rmouth_y)
    
    MISSING_FILES = [
        ("identity_CelebA.txt", "1"), # just filename id
        ("list_bbox_celeba.txt", "0 0 0 0"),
        ("list_landmarks_align_celeba.txt", "0 0 0 0 0 0 0 0 0 0")
    ]
    
    # We need the filenames. We can read them from list_eval_partition.txt
    if not os.path.exists(EVAL_TXT):
        print("Cannot create dummy files without list_eval_partition.txt")
        return

    print("Reading filenames for dummy generation...")
    with open(EVAL_TXT, "r") as f:
        lines = f.readlines()
        
    filenames = [l.split()[0] for l in lines]
    
    if len(filenames) != num_imgs:
        print(f"Warning: Num images mismatch {len(filenames)} vs {num_imgs}")

    for fname, dummy_val in MISSING_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            print(f"Skipping {fname}, exists.")
            continue
            
        print(f"Generating dummy {fname}...")
        with open(fpath, "w") as f:
            if fname.startswith("list_"):
                # these have headers
                f.write(f"{len(filenames)}\n")
                if "bbox" in fname:
                    f.write("image_id x_1 y_1 width height\n")
                elif "landmarks" in fname:
                    f.write("image_id leye_x leye_y reye_x reye_y nose_x nose_y lmouth_x lmouth_y rmouth_x rmouth_y\n")
            
            for name in filenames:
                f.write(f"{name} {dummy_val}\n")
                
    print("Dummy files generated.")

if __name__ == "__main__":
    print("Starting conversion...")
    try:
        convert_eval()
        convert_attr()
        
        # We need to know num images for dummy files
        # We can just count lines in eval or pass it if returned
        # Reading eval again is safe
        with open(EVAL_TXT, 'r') as f:
            num_imgs = sum(1 for _ in f)
        
        create_dummy_files(num_imgs)
        
        print("\nAll done! You should now have all required .txt files in data/celeba/")
    except Exception as e:
        print(f"An error occurred: {e}")
