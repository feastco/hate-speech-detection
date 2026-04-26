"""
Download dataset dari GitHub repository.
Jalankan di Anaconda Prompt setelah setup_env.bat:
    conda activate hate-speech
    cd D:\www\hate-speech
    python download_dataset.py
"""
import os
import urllib.request
import ssl

# URL dataset dari GitHub (Ibrohim & Budi, 2019)
DATASET_URL = (
    "https://raw.githubusercontent.com/okkyibrohim/"
    "id-multi-label-hate-speech-and-abusive-language-detection/"
    "master/re_dataset.csv"
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'data', 'raw')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'dataset_tweet.csv')


def download():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        size_kb = os.path.getsize(OUTPUT_FILE) / 1024
        print(f"Dataset sudah ada: {OUTPUT_FILE} ({size_kb:.1f} KB)")
        resp = input("Download ulang? (y/n): ").strip().lower()
        if resp != 'y':
            print("Batal. Menggunakan file yang sudah ada.")
            return

    print(f"Downloading dari GitHub...")
    print(f"URL: {DATASET_URL}")
    print(f"Target: {OUTPUT_FILE}")

    # Handle SSL untuk Windows
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        urllib.request.urlretrieve(DATASET_URL, OUTPUT_FILE)
        size_kb = os.path.getsize(OUTPUT_FILE) / 1024
        print(f"\n✅ Download selesai! ({size_kb:.1f} KB)")

        # Quick verification
        import pandas as pd
        df = pd.read_csv(OUTPUT_FILE, encoding='latin-1')
        print(f"   Jumlah baris : {len(df)}")
        print(f"   Kolom        : {df.columns.tolist()}")

    except Exception as e:
        print(f"\n❌ Gagal download: {e}")
        print("\nAlternatif manual:")
        print("  1. Buka: https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection")
        print("  2. Download file 're_dataset.csv'")
        print(f"  3. Simpan ke: {OUTPUT_FILE}")
        print("\nAtau via Kaggle:")
        print("  pip install kaggle")
        print("  kaggle datasets download -d ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text")


if __name__ == '__main__':
    download()
