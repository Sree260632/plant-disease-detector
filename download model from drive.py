import os
import gdown

def download_model():
    model_path = "model/trained_model.keras"
    if not os.path.exists(model_path):
        file_id = "10kq0xS3WKsaz1YHiQ64Rjn2Q-xhHk4kt"
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs("model", exist_ok=True)
        gdown.download(url, model_path, quiet=False)
    return model_path