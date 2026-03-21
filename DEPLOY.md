# 🚀 Deploy Hnhat AI lên Web

## Cấu trúc thư mục cần có:
```
AI/
├── Hnhatai.py          ← file chính
├── hnhatai.png         ← icon của bạn
├── requirements.txt    ← dependencies
├── Procfile            ← cho Railway/Render
└── .env.example        ← mẫu env vars
```

---

## ☁️ Cách 1: Railway (Miễn phí, dễ nhất)

1. Tạo repo GitHub, push toàn bộ thư mục AI lên
2. Vào https://railway.app → New Project → Deploy from GitHub
3. Chọn repo → Railway tự build
4. Vào **Variables** tab → thêm:
   - `GROQ_API_KEY`  = gsk_xxx
   - `GEMINI_API_KEY` = AIza_xxx
   - `SECRET_KEY`    = chuỗi random bất kỳ
5. Domain tự động: `https://xxx.railway.app`

---

## ☁️ Cách 2: Render (Miễn phí)

1. Push lên GitHub
2. Vào https://render.com → New Web Service
3. Connect GitHub repo
4. Build: `pip install -r requirements.txt`
5. Start: `gunicorn Hnhatai:app --bind 0.0.0.0:$PORT`
6. Add Environment Variables:
   - `GROQ_API_KEY`, `GEMINI_API_KEY`, `SECRET_KEY`

---

## 🖥️ Cách 3: VPS Ubuntu

```bash
# Cài dependencies
sudo apt update && sudo apt install python3-pip nginx certbot -y
pip3 install -r requirements.txt

# Chạy với systemd
sudo nano /etc/systemd/system/hnhatai.service
```

```ini
[Unit]
Description=Hnhat AI
After=network.target

[Service]
User=root
WorkingDirectory=/root/AI
ExecStart=gunicorn Hnhatai:app --bind 127.0.0.1:5000 --workers 2
Restart=always
Environment=GROQ_API_KEY=gsk_xxx
Environment=GEMINI_API_KEY=AIza_xxx
Environment=SECRET_KEY=your-secret

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable hnhatai && sudo systemctl start hnhatai

# Nginx reverse proxy
sudo nano /etc/nginx/sites-available/hnhatai
```

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;  # Quan trọng cho streaming!
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/hnhatai /etc/nginx/sites-enabled/
sudo certbot --nginx -d yourdomain.com  # SSL miễn phí
sudo systemctl restart nginx
```

---

## ⚠️ Lưu ý bảo mật

- KHÔNG đẩy file `.env` lên GitHub
- KHÔNG để API key trong code
- Thêm `.env` vào `.gitignore`
- Dùng `SECRET_KEY` ngẫu nhiên (https://randomkeygen.com)
