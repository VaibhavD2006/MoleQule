# MoleQule Deployment Guide

This guide will help you deploy the MoleQule application on Netlify (frontend) and a backend service.

## Prerequisites

- GitHub account
- Netlify account
- Railway/Render/Heroku account (for backend)
- Python 3.8+ (for local testing)

## Backend Deployment

### Option 1: Railway (Recommended)

1. **Fork/Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd molequle/backend
   ```

2. **Create Railway account and project**
   - Go to [Railway](https://railway.app)
   - Sign up with GitHub
   - Create new project
   - Choose "Deploy from GitHub repo"

3. **Configure environment variables in Railway**
   ```
   PORT=8000
   DATABASE_URL=sqlite:///./molequle.db
   CORS_ORIGINS=https://your-netlify-app.netlify.app
   ```

4. **Deploy**
   - Railway will automatically detect the Python app
   - It will install dependencies from `requirements.txt`
   - The app will be available at `https://your-app-name.railway.app`

### Option 2: Render

1. **Create Render account**
   - Go to [Render](https://render.com)
   - Sign up with GitHub

2. **Create new Web Service**
   - Connect your GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Configure environment variables**
   ```
   PORT=8000
   DATABASE_URL=sqlite:///./molequle.db
   CORS_ORIGINS=https://your-netlify-app.netlify.app
   ```

## Frontend Deployment (Netlify)

### Step 1: Prepare the Frontend

1. **Update API URL**
   - In `molequle/frontend/next.config.js`, update the `NEXT_PUBLIC_API_URL` to your backend URL
   - Or set it as an environment variable in Netlify

2. **Test locally**
   ```bash
   cd molequle/frontend
   npm install
   npm run build
   ```

### Step 2: Deploy to Netlify

1. **Connect to GitHub**
   - Go to [Netlify](https://netlify.com)
   - Sign up with GitHub
   - Click "New site from Git"

2. **Configure build settings**
   - Repository: Your GitHub repo
   - Base directory: `molequle/frontend`
   - Build command: `npm run build`
   - Publish directory: `out`

3. **Set environment variables**
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
   ```

4. **Deploy**
   - Click "Deploy site"
   - Netlify will build and deploy your site

## Environment Variables

### Backend (.env)
```
PORT=8000
DATABASE_URL=sqlite:///./molequle.db
CORS_ORIGINS=https://your-netlify-app.netlify.app
```

### Frontend (Netlify Environment Variables)
```
NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
```

## Post-Deployment

1. **Test the application**
   - Upload a molecule file
   - Check if results are generated
   - Verify downloads work

2. **Monitor logs**
   - Check Railway/Render logs for backend issues
   - Check Netlify build logs for frontend issues

3. **Set up custom domain (optional)**
   - Configure custom domain in Netlify
   - Update CORS settings in backend

## Troubleshooting

### Common Issues

1. **CORS errors**
   - Ensure backend CORS_ORIGINS includes your Netlify domain
   - Check that the API URL is correct

2. **Build failures**
   - Check Node.js version (use 18.x)
   - Verify all dependencies are installed

3. **API connection issues**
   - Verify backend is running
   - Check environment variables
   - Test API endpoints directly

### Support

If you encounter issues:
1. Check the logs in your deployment platform
2. Verify environment variables are set correctly
3. Test API endpoints using curl or Postman
4. Check the browser console for frontend errors

## Security Considerations

1. **Environment variables**
   - Never commit sensitive data to Git
   - Use environment variables for all configuration

2. **CORS**
   - Only allow necessary origins
   - Don't use wildcard (*) in production

3. **File uploads**
   - Validate file types and sizes
   - Consider implementing rate limiting

## Performance Optimization

1. **Frontend**
   - Enable gzip compression in Netlify
   - Optimize images and assets
   - Use CDN for static assets

2. **Backend**
   - Implement caching where appropriate
   - Optimize database queries
   - Consider using a production database (PostgreSQL)

## Monitoring

1. **Set up monitoring**
   - Use Railway/Render's built-in monitoring
   - Set up error tracking (Sentry)
   - Monitor API response times

2. **Logs**
   - Regularly check application logs
   - Set up log aggregation if needed 