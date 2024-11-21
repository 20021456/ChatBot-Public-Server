# Cài đặt chatbot
## Môi trường ảo
Tải virtualenv (Windows)
```
py -m pip install --user virtualenv
```
Linux
```
python3 -m pip install --user virtualenv
```
Chạy lần lượt 2 câu lệnh sau:
```
python -m venv .venv
```
```
.venv/Scripts/Activate.ps1
```
hoặc
```
source .venv/bin/activate
```
## Cài đặt các dependencies
```
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```
```
pip install -r requirements.txt
```
## Thêm API Key
## Chạy chương trình
```
streamlit run app.py
```