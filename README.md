# 🦙 รัน Model (GGUF) บน RunPod ด้วย llama.cpp + Web Chat

Deploy POD  with pytorch 2.12 +  and create port 8000 or 8080 fot web chat

use terminal

cd workspace

put file run.sh in workspace 

````
#chmod +x gemma-4-31B-it-ARA-heretic.Q6_K.sh
#./gemma-4-31B-it-ARA-heretic.Q6_K.sh
````

````
(ถ้าอยากเร็วขึ้น 2–3 เท่า)
แก้บรรทัดเดียวใน Script ให้ตรงกับ GPU ที่ใช้จริง:
# ถ้าใช้ A40
CUDA_ARCH="86"

# ถ้าใช้ L40 / L40S
CUDA_ARCH="89"

# ถ้าใช้ H100
CUDA_ARCH="90"
จะลดเวลา build จาก ~30 นาที เหลือ ~10–12 นาที
````
