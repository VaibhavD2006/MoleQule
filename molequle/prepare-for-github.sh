#!/bin/bash

# Prepare MoleQule for GitHub upload
# This script ensures all necessary files are included for Render deployment

echo "🔧 Preparing MoleQule for GitHub upload..."
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "backend/main.py" ] || [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: Please run this script from the molequle directory"
    exit 1
fi

echo "📋 Checking essential files..."

# Check backend files
backend_files=(
    "backend/main.py"
    "backend/requirements.txt"
    "backend/Procfile"
    "backend/runtime.txt"
    "backend/app/__init__.py"
    "backend/app/api/jobs.py"
    "backend/app/models/database.py"
)

missing_files=()
for file in "${backend_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ Missing essential files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo "✅ All essential files found"

# Create uploads directory if it doesn't exist
if [ ! -d "backend/uploads" ]; then
    echo "📁 Creating uploads directory..."
    mkdir -p backend/uploads
    echo "# Upload directory for molecular files" > backend/uploads/README.md
fi

# Create .gitkeep to ensure uploads directory is tracked
touch backend/uploads/.gitkeep

echo ""
echo "📝 Files to commit to GitHub:"
echo "=============================="
echo ""
echo "✅ Essential Backend Files:"
echo "   - backend/main.py"
echo "   - backend/requirements.txt"
echo "   - backend/Procfile"
echo "   - backend/runtime.txt"
echo "   - backend/env.example"
echo "   - backend/app/ (entire directory)"
echo "   - backend/uploads/ (directory)"
echo ""
echo "✅ Configuration Files:"
echo "   - .gitignore"
echo "   - DEPLOYMENT.md"
echo "   - README_DEPLOYMENT.md"
echo ""
echo "✅ Frontend Files (for Netlify):"
echo "   - frontend/ (entire directory)"
echo ""
echo "❌ Files NOT to commit:"
echo "   - .env files (contain sensitive data)"
echo "   - *.db files (database files)"
echo "   - node_modules/ (npm dependencies)"
echo "   - __pycache__/ (Python cache)"
echo "   - uploads/*.mol, *.smi, *.xyz (uploaded files)"
echo ""

echo "🚀 Next steps:"
echo "1. Initialize git repository (if not already done):"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial commit for Render deployment'"
echo ""
echo "2. Create GitHub repository and push:"
echo "   git remote add origin https://github.com/yourusername/your-repo-name.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Deploy on Render:"
echo "   - Go to https://render.com"
echo "   - Connect your GitHub repository"
echo "   - Create new Web Service"
echo "   - Set build command: pip install -r requirements.txt"
echo "   - Set start command: uvicorn main:app --host 0.0.0.0 --port \$PORT"
echo "   - Set environment variables (see env.example)"
echo ""
echo "📖 See DEPLOYMENT.md for detailed instructions" 