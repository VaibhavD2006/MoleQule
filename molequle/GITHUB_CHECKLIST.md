# GitHub Upload Checklist for Render Deployment

## ✅ **Files to Upload to GitHub**

### **Backend Files (Required for Render)**
```
molequle/backend/
├── main.py                    # ✅ REQUIRED - FastAPI app entry point
├── requirements.txt           # ✅ REQUIRED - Python dependencies
├── Procfile                  # ✅ REQUIRED - Render deployment config
├── runtime.txt               # ✅ REQUIRED - Python version
├── env.example               # ✅ REQUIRED - Environment variables template
├── app/                      # ✅ REQUIRED - Application code
│   ├── __init__.py
│   ├── api/
│   │   └── jobs.py          # API endpoints
│   ├── models/
│   │   └── database.py      # Database models
│   └── services/
│       └── [service files]  # Business logic
└── uploads/                  # ✅ REQUIRED - Upload directory
    ├── .gitkeep             # Keep directory in git
    └── README.md            # Directory description
```

### **Configuration Files**
```
molequle/
├── .gitignore               # ✅ REQUIRED - Git ignore rules
├── DEPLOYMENT.md            # ✅ RECOMMENDED - Deployment guide
├── README_DEPLOYMENT.md     # ✅ RECOMMENDED - Quick guide
└── prepare-for-github.sh    # ✅ OPTIONAL - Helper script
```

### **Frontend Files (For Netlify)**
```
molequle/frontend/           # ✅ REQUIRED - Next.js app
├── package.json
├── next.config.js
├── netlify.toml
├── src/
│   ├── pages/
│   ├── components/
│   └── styles/
└── public/
```

## ❌ **Files NOT to Upload**

### **Sensitive Files**
- `.env` files (contain API keys, passwords)
- `*.db` files (database files)
- `*.sqlite` files

### **Build Artifacts**
- `node_modules/` (npm dependencies)
- `__pycache__/` (Python cache)
- `*.pyc` files
- `out/` directory (Next.js build output)
- `.next/` directory

### **Uploaded Files**
- `uploads/*.mol` (molecular files)
- `uploads/*.smi` (SMILES files)
- `uploads/*.xyz` (XYZ files)

## 🚀 **Quick Upload Commands**

```bash
# 1. Initialize git (if not already done)
git init

# 2. Add all files (respects .gitignore)
git add .

# 3. Commit
git commit -m "Initial commit for Render deployment"

# 4. Create GitHub repository and push
git remote add origin https://github.com/yourusername/your-repo-name.git
git branch -M main
git push -u origin main
```

## 🔧 **Render Deployment Settings**

Once uploaded to GitHub, configure Render with:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Root Directory**: `molequle/backend`

### **Environment Variables to Set in Render**
```
PORT=8000
CORS_ORIGINS=https://your-netlify-app.netlify.app
DATABASE_URL=sqlite:///./molequle.db
```

## ✅ **Verification Checklist**

Before pushing to GitHub, ensure:

- [ ] All backend files are present
- [ ] `.gitignore` excludes sensitive files
- [ ] `requirements.txt` has all dependencies
- [ ] `Procfile` is correctly formatted
- [ ] `runtime.txt` specifies Python version
- [ ] No `.env` files are included
- [ ] No database files are included
- [ ] Uploads directory exists with `.gitkeep`

## 🆘 **Troubleshooting**

If deployment fails:

1. **Check Render logs** for build errors
2. **Verify all files** are in the repository
3. **Check environment variables** are set correctly
4. **Ensure Python version** matches `runtime.txt`
5. **Verify dependencies** in `requirements.txt` 