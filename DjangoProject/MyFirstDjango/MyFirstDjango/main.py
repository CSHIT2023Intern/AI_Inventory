from fastapi import FastAPI, UploadFile, File
import torch
import cv2
import numpy as np
from typing import List
from .detect import detectapi

app = FastAPI()

# 创建 detectapi 实例
detector = detectapi(weights='static/07211657_best.pt', img_size=640)

@app.post("/detect/")
async def detect_objects(files: List[UploadFile]):
    results = []

    for file in files:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0

        with torch.no_grad():
            detection_results, _ = detector.detect([image_tensor])

         # 处理检测结果并构建响应
        # 你可以根据需要自行调整处理逻辑，这里只是一个示例
        vial_count = 0
        ampoule_count = 0
        for _, result_txt in detection_results:
            for cls, _, _ in result_txt:
                if cls == 0:  # Vial
                    vial_count += 1
                elif cls == 1:  # Ampoule
                    ampoule_count += 1

        total_count = vial_count + ampoule_count

        result = {
            "vial_count": vial_count,
            "ampoule_count": ampoule_count,
            "total_count": total_count
        }
        results.append(result)

    return results
