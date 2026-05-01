# 🚀 Deployment Guide: OncoGraph X

Follow these steps to deploy **OncoGraph X** for free using **Render** (Backend) and **Vercel** (Frontend).

---

## 1. Prepare your Repository
1. Initialize a Git repository in this folder: `git init`
2. Create a repository on GitHub (e.g., `oncograph-x`).
3. Commit and push your code:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

---

## 2. Deploy the Backend (Render)
1. Log in to [Render](https://render.com/).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. Configure the service:
   - **Name**: `oncograph-x-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`
5. Click **Deploy Web Service**.
6. **Note the URL**: Once deployed, Render will give you a URL (e.g., `https://oncograph-x-api.onrender.com`). **Copy this URL.**

---

## 3. Deploy the Frontend (Vercel)
1. Log in to [Vercel](https://vercel.com/).
2. Click **Add New** and select **Project**.
3. Import your GitHub repository.
4. Configure the Project:
   - **Root Directory**: Select `frontend` (this is critical).
   - **Framework Preset**: `Next.js`.
5. **Environment Variables**:
   - Add a new variable:
     - **Key**: `NEXT_PUBLIC_API_URL`
     - **Value**: `<The Render URL you copied in Step 2>` (e.g., `https://oncograph-x-api.onrender.com`)
6. Click **Deploy**.

---

## 4. Final Verification
- Once Vercel finishes building, visit your new dashboard URL.
- It should now be fetching data from your live Render API!

---

### 💡 Pro Tips for Free Tiers
- **Render Free Tier**: The backend will "spin down" after 15 minutes of inactivity. The first request after a break might take ~30 seconds to wake up the server.
- **Vercel**: Next.js deployments are blazing fast and will stay active 24/7.
