# 🦙 รัน Model (GGUF) บน RunPod ด้วย llama.cpp + Web Chat

คู่มือแบบทีละขั้นตอน สำหรับรันโมเดล
บน **RunPod** โดยใช้ **llama.cpp** เป็น inference engine
และเปิด **หน้าเว็บสำหรับแชททดสอบ** (ทั้ง UI ในตัวของ llama-server และ Open WebUI)

---

## 📑 สารบัญ
- [ภาพรวม & ทำไมต้อง llama.cpp](#-ภาพรวม--ทำไมต้อง-llamacpp)
- [สิ่งที่ต้องเตรียม (Requirements)](#-สิ่งที่ต้องเตรียม-requirements)
- [0. สร้าง Pod บน RunPod](#0-สร้าง-pod-บน-runpod)
- [1. โหลดโมเดล (GGUF)](#1-โหลดโมเดล-gguf)
- [2. ติดตั้ง llama.cpp (build from source)](#2-ติดตั้ง-llamacpp-build-from-source)
- [3. รัน llama-server](#3-รัน-llama-server)
- [4. แชทผ่านหน้าเว็บ](#4-แชทผ่านหน้าเว็บ)
- [5. ค่า Sampling (โปรไฟล์)](#5-ค่า-sampling-โปรไฟล์)
- [Troubleshooting](#-troubleshooting)
- [Checklist สรุป](#-checklist-สรุป)

---

## 🧭 ภาพรวม & ทำไมต้อง llama.cpp

| Engine | สถานะกับโมเดลนี้ | หมายเหตุ |
|---|---|---|
| **llama.cpp (latest)** | ✅ รองรับชัวร์ + เร็วสุด | เป็น "ต้นน้ำ" ที่ทำ operator ใหม่เอง |
| Ollama (latest) | 🟡 ใช้ได้แต่ตามหลัง/มักช้ากว่า | image ค้างเก่า = โหลดไม่ได้เลย |
| koboldcpp | ⚠️ สถานะ qwen3.5 ยังไม่นิ่ง | — |
| vLLM / SGLang | 🔴 ไม่เหมาะกับ GGUF การ์ดเดียว | เก่งกับ safetensors + หลาย GPU |

> **สรุป:** ใช้ **llama.cpp build from source** → ได้เวอร์ชันล่าสุดที่รองรับ
> `qwen35`/DeltaNet แน่นอน และอ่าน GGUF โดยตรง ไม่ต้องแปลงไฟล์

---

## ⚙️ สิ่งที่ต้องเตรียม (Requirements)

- **GPU:** VRAM **48 GB ขึ้นไป** (RTX 6000 Ada / L40S / A6000 / A40)
  - น้ำหนักโมเดล Q8_0 ≈ 30.2 GB *อย่างเดียว* ยังไม่รวม KV cache → 24 GB **ไม่พอ**
  - ถ้าอยากเผื่อ context ยาวมาก ๆ ค่อยใช้ 80 GB (A100/H100)
- **Disk:** Volume **80–100 GB** (โหลดโมเดลไว้ `/workspace` เพื่อไม่ให้หายตอน Stop Pod)
  + Container Disk 30–40 GB
- **Template:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
  - มี CUDA 12.8 + build tools ครบ → build llama.cpp from source ได้
  - (PyTorch ที่ติดมา *ไม่ได้ใช้* เพราะ llama.cpp ไม่พึ่ง PyTorch — มีไว้ก็ไม่เป็นไร)

---

## 0. สร้าง Pod บน RunPod

1. เข้า RunPod → **Deploy** → เลือก template
   `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
2. เลือก GPU **48 GB+**
3. ตั้ง **Volume Disk 80–100 GB** (mount ที่ `/workspace`)
4. Deploy → รอขึ้น Running → กด **Connect** → เปิด **Terminal** (หรือ Web Terminal)

---

## 1. โหลดโมเดล (GGUF)

### เลือกไฟล์: regular vs MTP

- **regular Q8_0** (ไม่มีคำว่า `MTP`) → โหลดแล้วแชทได้เลย **ชัวร์ที่สุด** ✅ *แนะนำสำหรับเทสแชท*
- **MTP Q8_0** (มี `-NEO-MTP-`) → โมเดลปกติ + หัว *Multi-Token Prediction*
  สำหรับเร่งความเร็ว (speculative decoding) **คุณภาพ identical** กับ regular
  แต่ต้องเปิด flag เพิ่มถึงจะได้ speedup → ตัวแปรเยอะกว่า

> เริ่มจาก **regular** ก่อน ถ้าอยากลองของใหม่ค่อยโหลด MTP

### 1.1 โหลดด้วย `wget` (ง่ายสุด — ใช้ได้ใน Jupyter/Terminal)

```bash
mkdir -p /workspace/models && cd /workspace/models

# wget (ตามด้วย link model .gguf)
```

> ⚠️ ถ้า regular ขึ้น `404 Not Found` แปลว่าชื่อไฟล์จริงต่างนิดหน่อย →
> เปิดหน้า repo ในเบราว์เซอร์ → หาไฟล์ `*Q8_0.gguf` ที่ **ไม่มีคำว่า MTP** →
> คลิกขวา **Copy link address** → เอามาวางต่อหลัง `wget` แทน (ชัวร์ 100%)

**เช็กว่าโหลดครบ:**
```bash
ls -lh /workspace/models/*.gguf   # ต้องเห็น ~30.2 GB
```

---

## 2. ติดตั้ง llama.cpp (build from source)

> ทำ *ขนาน* กับการดาวน์โหลดโมเดลได้ (เปิด terminal อีกแท็บ) เพื่อประหยัดเวลา

```bash
apt-get update && apt-get install -y cmake build-essential git

cd /workspace
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# เช็กว่า build สำเร็จ + เห็น CUDA
./build/bin/llama-server --version
```

> ใช้เวลาไม่กี่นาทีบน Pod GPU และได้ llama.cpp **latest** ที่รองรับ `qwen35` แน่นอน

---

## 3. รัน llama-server

```bash
cd /workspace/llama.cpp

./build/bin/llama-server \
  -m /workspace/models/ชื่อmodel ที่โหลดมา \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 16384 \
  --jinja \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0
```

**ความหมาย flag:**

| Flag | ความหมาย |
|---|---|
| `-m ...` | path ไปยังไฟล์ GGUF (เปลี่ยนชื่อไฟล์ถ้าใช้ MTP) |
| `--host 0.0.0.0` | เปิดให้ proxy ของ RunPod เข้าถึงได้ |
| `--port 8080` | พอร์ตของ server (ทั้ง UI และ API) |
| `-ngl 99` | ยัดทุก layer ลง GPU (48 GB+ เอาอยู่สำหรับ 30 GB) |
| `-c 16384` | จำกัด context กัน OOM (โมเดลรองรับ 256K แต่การ์ดเดียวต้องจำกัด; เหลือเฟือค่อยดัน `32768`) |
| `--jinja` | ใช้ **chat template qwen3 ที่ฝังใน GGUF** → *สำคัญ* อย่าเขียน template เอง |
| sampling | โหมด *Thinking – general* ตามที่ผู้แต่งโมเดลแนะนำ |

> ถ้าใช้ไฟล์ **MTP** และอยากได้ speedup: เพิ่ม `--spec-type draft-mtp`
> (พร้อมค่า draft) เข้าไป — แต่ถ้าโหลดแล้ว error เรื่อง MTP ให้ถอยมาใช้ regular

---

## 4. แชทผ่านหน้าเว็บ

### 4.1 วิธีเร็วสุด: Chat UI ในตัวของ llama-server ✅

`llama-server` ฝัง chat UI มาให้ที่ root URL เลย ไม่ต้องลงอะไรเพิ่ม

1. ในหน้า Pod ของ RunPod → ดูส่วน **Connect / Proxy**
2. กดปุ่ม **HTTPS link** ที่ชี้มา port **`8080`**
3. เบราว์เซอร์จะเปิดหน้า **llama.cpp chat UI** → เลือก/พิมพ์เทสโมเดลได้ทันที

### 4.2 วิธีสวย/ครบฟีเจอร์: ต่อ Open WebUI 

`llama-server` พูด **OpenAI-compatible API** (`/v1`) ดังนั้นต่อ Open WebUI ทับได้เลย

> ⚠️ Open WebUI ใช้ port `8080` เป็นค่าเริ่มต้น **ชนกับ llama-server** →
> จึงให้ Open WebUI ไปใช้ port **`7860`** แทน

**ติดตั้ง (ใน terminal อีกแท็บ — ไม่ต้องหยุด llama-server):**

```bash
# สร้าง virtual env แยก ไม่ให้ชนกับระบบ
cd /workspace
python -m venv owui-venv
source owui-venv/bin/activate
pip install -U pip
pip install open-webui
```

**รัน Open WebUI ชี้ไปที่ llama-server:**

```bash
source /workspace/owui-venv/bin/activate

export OPENAI_API_BASE_URL="http://127.0.0.1:8080/v1"   # llama-server
export OPENAI_API_KEY="ollama"                            # ใส่ค่าอะไรก็ได้
export ENABLE_OLLAMA_API="false"                          # กันมันเรียก ollama:11434 แล้วพัง
export WEBUI_SECRET_KEY="my-secret-change-me"

open-webui serve --host 0.0.0.0 --port 7860
```

**เปิดหน้าเว็บ:**

1. ในหน้า Pod ของ RunPod → กดปุ่ม **HTTPS link** ที่ชี้มา port **`7860`**
2. ครั้งแรกให้สร้างบัญชี admin (ล็อกไว้ในเครื่องคุณเอง)
3. เข้าไปจะเห็นโมเดลจาก llama-server → เลือกแล้ว **แชททดสอบ** ได้เลย
   (ปรับ temp / top_p / top_k / context ได้ในหน้า Settings ของแต่ละแชท)

---

## 5. ค่า Sampling (โปรไฟล์)

โมเดลนี้ **คิด (thinking) เป็นค่าเริ่มต้น** → คำตอบจะขึ้นบล็อก `<think>...</think>`
ก่อนตอบจริง (WebUI แสดงผลได้ปกติ) เลือกโปรไฟล์ตามงาน:

| โปรไฟล์ | temp | top_p | top_k | min_p | presence_penalty | repeat_penalty |
|---|---|---|---|---|---|---|
| **Thinking – ทั่วไป** | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| **Thinking – โค้ด/WebDev** | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| **Instruct (ไม่คิด)** | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

> สำหรับ **MTP**: ผู้แต่งระบุให้ keep `temp ≤ 1` และ `repeat_penalty = 1`

---

## 🛠 Troubleshooting

- **`unknown architecture` / โหลดโมเดลไม่ขึ้น**
  → llama.cpp เก่าไป → `cd /workspace/llama.cpp && git pull && cmake --build build --config Release -j$(nproc)` แล้วรันใหม่
- **OOM / CUDA out of memory**
  → ลด `-c` เหลือ `8192` หรือ `4096`; เช็กว่า `-ngl 99` (อย่าเหลือ layer บน CPU โดยไม่ตั้งใจ)
- **wget/hf ได้ไฟล์ไม่ครบขนาด**
  → รันซ้ำด้วย `wget -c <url>` (resume) แล้ว `ls -lh` เช็กให้ครบ ~30.2 GB
- **`404 Not Found` ตอน wget**
  → ชื่อไฟล์ต่างจากที่เดา → copy link จริงจากหน้า repo (คลิกขวา Copy link address)
- **เปิด proxy port แล้วหน้าเว็บไม่ขึ้น**
  → เช็กว่าสั่ง `--host 0.0.0.0` แล้ว; เช็ก log ว่า server listen ที่ port นั้นจริง;
  → Open WebUI ต้องใช้ port คนละตัวกับ llama-server (8080 vs 7860)
- **Open WebUI หาโมเดลไม่เจอ**
  → เช็ก `curl http://127.0.0.1:8080/v1/models` ต้องตอบกลับ;
  → เช็ก env `OPENAI_API_BASE_URL` และ `ENABLE_OLLAMA_API=false`
- **Vision/รูปภาพ**
  → ข้ามไปก่อนสำหรับการเทสแชท (ต้องโหลด `mmproj` แยก และการรองรับใน Ollama/llama.cpp ยังไม่สมบูรณ์)

---

## ✅ Checklist สรุป

- [ ] สร้าง Pod: template pytorch cu128 + GPU 48 GB+ + Volume 80–100 GB
- [ ] โหลด GGUF ครบ ~30.2 GB ที่ `/workspace/models` (เริ่มจาก regular)
- [ ] build llama.cpp from source สำเร็จ (`llama-server --version` เห็น CUDA)
- [ ] รัน `llama-server` ด้วย `--jinja`, `-ngl 99`, `-c 16384`, `--host 0.0.0.0`
- [ ] เปิด proxy port `8080` → เทส chat UI ในตัว
- [ ] (ถ้าต้องการ) ลง Open WebUI ที่ port `7860` ชี้ `/v1` ของ llama-server
- [ ] ตั้ง sampling ตามโปรไฟล์ที่ต้องการ

---

## 🔗 ลิงก์อ้างอิง

- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- llama.cpp releases (prebuilt): <https://github.com/ggml-org/llama.cpp/releases>
- Open WebUI: <https://github.com/open-webui/open-webui>
