# ParamkaraCorp-HR

AI-powered HR verification agent. Upload any document — it auto-detects and analyzes.

## Features
- JD vs Resume skill matching
- Offer letter fraud detection (CIN, GST checksum)
- Salary slip authenticity (PF math, CTC inflation)
- Employment history / EPFO cross-check

## Deploy to Netlify (5 minutes)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "ParamkaraCorp-HR"
git remote add origin https://github.com/YOUR_USERNAME/paramkaracorp-hr.git
git push -u origin main
```

### 2. Deploy on Netlify
1. Go to https://netlify.com → New site from Git
2. Connect your GitHub repo
3. Build settings are auto-detected from netlify.toml

### 3. Set Environment Variable
In Netlify Dashboard → Site settings → Environment variables:
```
GROQ_API_KEY = your_groq_api_key_here
```
Get free key at: https://console.groq.com

### 4. Done!
Your site will be live at `https://your-site.netlify.app`

## Local development
```bash
npm install -g netlify-cli
netlify dev
```
Open http://localhost:8888
