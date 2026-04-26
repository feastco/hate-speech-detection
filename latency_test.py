import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

url = "http://127.0.0.1:8000/predict"
# Gunakan teks campuran (beberapa hate speech, beberapa abusive, beberapa normal)
texts = [
    "Dasar bego, gitu aja gak becus kerja!",
    "Usir saja orang-orang itu dari sini, mereka cuma bikin masalah!",
    "Selamat pagi semuanya, semoga hari ini menyenangkan.",
    "Kamu ini bodoh sekali, tidak usah sok tahu.",
    "Jangan kasih suara buat dia, dia pengkhianat."
]

session = requests.Session()

def send_request(i, model_type):
    text = texts[i % len(texts)]
    payload = {
        "text": text,
        "model_type": model_type
    }
    start_time = time.time()
    try:
        response = session.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return None
    end_time = time.time()
    # Return latensi dalam milidetik
    return (end_time - start_time) * 1000

def run_test(model_type, n_requests=500, concurrency=1):
    latencies = []
    
    # Pemanasan (Warmup) agar model/cache siap
    for i in range(5):
        send_request(i, model_type)
        
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda i: send_request(i, model_type), range(n_requests)))
        
    latencies = [r for r in results if r is not None]
    failed = n_requests - len(latencies)
    
    if not latencies:
        print(f"{model_type.upper():<10} | SEMUA REQUEST GAGAL")
        return

    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    print(f"{model_type.upper():<10} | Avg: {avg_latency:>6.2f} ms | Med: {median_latency:>6.2f} ms | Min: {min_latency:>6.2f} ms | Max: {max_latency:>6.2f} ms | Failed: {failed}")

if __name__ == "__main__":
    print("-" * 80)
    print("HASIL PENGUJIAN LATENSI PER MODEL (n=500 request)")
    print("-" * 80)
    models = ["svm", "mnb", "cnb", "indobert"]
    for model in models:
        run_test(model, n_requests=500, concurrency=1)
    print("-" * 80)
