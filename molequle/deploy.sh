#!/bin/bash

# MoleQule Deployment Script
# This script helps prepare the application for deployment

echo "🚀 MoleQule Deployment Script"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "backend/main.py" ] || [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: Please run this script from the molequle directory"
    exit 1
fi

echo "📋 Checking prerequisites..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+"
    exit 1
fi

echo "✅ Prerequisites check passed"

echo ""
echo "🔧 Setting up frontend..."

# Install frontend dependencies
cd frontend
npm install

# Build frontend
echo "📦 Building frontend..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Frontend build successful"
else
    echo "❌ Frontend build failed"
    exit 1
fi

cd ..

echo ""
echo "🐍 Setting up backend..."

# Install backend dependencies
cd backend
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Backend dependencies installed"
else
    echo "❌ Backend dependencies installation failed"
    exit 1
fi

cd ..

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Deploy backend to Railway/Render:"
echo "   - Go to https://railway.app or https://render.com"
echo "   - Connect your GitHub repository"
echo "   - Set environment variables (see DEPLOYMENT.md)"
echo ""
echo "2. Deploy frontend to Netlify:"
echo "   - Go to https://netlify.com"
echo "   - Connect your GitHub repository"
echo "   - Set base directory to 'molequle/frontend'"
echo "   - Set build command to 'npm run build'"
echo "   - Set publish directory to 'out'"
echo "   - Set environment variable NEXT_PUBLIC_API_URL to your backend URL"
echo ""
echo "📖 See DEPLOYMENT.md for detailed instructions" 