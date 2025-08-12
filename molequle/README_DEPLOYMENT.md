# Quick Deployment Guide

## 🚀 Deploy MoleQule to Netlify

### Prerequisites
- GitHub repository with your code
- Netlify account
- Railway/Render account (for backend)

### Step 1: Deploy Backend

1. **Railway (Recommended)**
   - Go to [Railway](https://railway.app)
   - Connect your GitHub repo
   - Set environment variables:
     ```
     PORT=8000
     CORS_ORIGINS=https://your-netlify-app.netlify.app
     ```
   - Deploy and note the URL (e.g., `https://your-app.railway.app`)

2. **Render (Alternative)**
   - Go to [Render](https://render.com)
   - Create new Web Service
   - Connect your GitHub repo
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Step 2: Deploy Frontend to Netlify

1. **Connect to GitHub**
   - Go to [Netlify](https://netlify.com)
   - Click "New site from Git"
   - Choose your GitHub repository

2. **Configure Build Settings**
   - Base directory: `molequle/frontend`
   - Build command: `npm run build`
   - Publish directory: `out`

3. **Set Environment Variables**
   - Go to Site settings > Environment variables
   - Add: `NEXT_PUBLIC_API_URL` = `https://your-backend-url.railway.app`

4. **Deploy**
   - Click "Deploy site"
   - Wait for build to complete

### Step 3: Test Your Application

1. Visit your Netlify URL
2. Upload a molecule file (.mol, .smi, .xyz)
3. Check if results are generated
4. Verify downloads work

### Troubleshooting

- **CORS errors**: Ensure backend CORS_ORIGINS includes your Netlify domain
- **Build failures**: Check Node.js version (use 18.x)
- **API errors**: Verify backend is running and environment variables are set

### Support

See `DEPLOYMENT.md` for detailed instructions and troubleshooting. 