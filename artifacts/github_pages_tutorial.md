# Publishing the GridLock R2 Demo Map to GitHub Pages

This tutorial walks you through publishing `enforcement_priority_2024-03-18_09h.html`
(or any other HTML output) to GitHub Pages so it is accessible via a public URL —
no web server, no hosting account needed.

---

## Prerequisites

- The GitHub repository already exists (e.g. `https://github.com/<your-username>/GridLock-R2`)
- You have `git` installed and configured with your GitHub credentials
- The HTML file you want to publish is in `data/outputs/`

---

## Step 1 — Decide Which Branch and Folder to Use

GitHub Pages can serve from two locations. Choose **Option A** for simplicity:

| Option | Branch | Root Folder | Best For |
|:---|:---|:---|:---|
| **A — docs/ folder on main** ✅ (recommended) | `main` | `/docs` | Simple projects; everything stays in one branch |
| B — Dedicated gh-pages branch | `gh-pages` | `/` (root) | Large sites; keeps HTML files out of main branch history |

**We will use Option A throughout this tutorial.** The public URL will be:
```
https://<your-username>.github.io/GridLock-R2/
```

---

## Step 2 — Create the `/docs` Folder and Copy the HTML File

Run these commands from the root of the repository (inside `GridLock R2/`):

```powershell
# Create the docs folder if it doesn't exist
mkdir docs

# Copy your HTML output into docs/ and rename it to index.html
# index.html is what GitHub Pages serves at the root URL
Copy-Item "data\outputs\enforcement_priority_2024-03-18_09h.html" "docs\index.html"
```

> **Why rename to `index.html`?**
> GitHub Pages automatically serves `index.html` at the root URL. If you keep the original filename,
> the URL becomes `/enforcement_priority_2024-03-18_09h.html` instead of just `/`.
> For a clean demo link, rename to `index.html`.

If you want to publish **multiple HTML files** (e.g. one per hour), keep all original filenames and
create a simple `docs/index.html` that links to them.

---

## Step 3 — Add `/docs` to `.gitignore` Exclusions

Check your `.gitignore`. If it currently ignores `data/outputs/`, you need to make sure `docs/` is **not** ignored:

```powershell
# View current .gitignore rules
Get-Content .gitignore
```

If you see a line like `data/outputs/`, add an explicit un-ignore rule for docs:

```
# In .gitignore — add this if docs/ would otherwise be excluded
!docs/
!docs/**
```

The `docs/` folder must be tracked by git to be published.

---

## Step 4 — Commit and Push the HTML File

```powershell
# Stage the docs folder
git add docs/

# Commit
git commit -m "Add enforcement priority map to docs/ for GitHub Pages"

# Push to main
git push origin main
```

Verify it pushed:
```powershell
git log --oneline -3
```
You should see your commit at the top.

---

## Step 5 — Enable GitHub Pages in Repository Settings

1. Open your repository in a browser: `https://github.com/<your-username>/GridLock-R2`
2. Click **Settings** (top navigation bar of the repo, not your profile settings)
3. In the left sidebar, click **Pages** (under the "Code and automation" section)
4. Under **"Build and deployment"**:
   - **Source**: Select `Deploy from a branch`
   - **Branch**: Select `main`
   - **Folder**: Select `/docs`
5. Click **Save**

GitHub will show a banner: *"GitHub Pages source saved."*

---

## Step 6 — Get the Public URL

After saving, GitHub Pages will build and deploy your site. This takes **1–3 minutes** on the first deployment.

Refresh the **Pages** settings page. At the top you will see:

```
Your site is live at https://<your-username>.github.io/GridLock-R2/
```

Click the link to verify the map loads correctly. The interactive Folium map should render fully — it is entirely self-contained in the HTML file (no external data requests).

> [!NOTE]
> If the page shows a 404 on first load, wait 2 minutes and hard-refresh (`Ctrl + Shift + R`).
> GitHub Pages CDN propagation can take a few minutes on the first deploy.

---

## Step 7 — Updating the Map When the HTML File Changes

Every time you run the pipeline and generate a new output HTML, update the published page by repeating Steps 2 and 4:

```powershell
# Regenerate the HTML (example: new date/hour)
python -m src.data.pipeline --skip-features --skip-clustering --skip-training --date 2024-03-19 --hour 9

# Overwrite the published file
Copy-Item "data\outputs\enforcement_priority_2024-03-19_09h.html" "docs\index.html" -Force

# Commit and push
git add docs/index.html
git commit -m "Update demo map: 2024-03-19 09h enforcement priorities"
git push origin main
```

GitHub Pages will automatically re-deploy within 1–2 minutes of the push. No manual action in Settings is required after the first setup.

---

## Sharing Multiple HTML Outputs (Optional)

If you want to publish multiple maps (e.g. morning peak, evening peak, full-day schedule), create a
`docs/index.html` landing page that links to each file:

```html
<!-- docs/index.html (simple landing page) -->
<!DOCTYPE html>
<html>
<head><title>GridLock R2 — Demo Maps</title></head>
<body>
  <h1>GridLock R2 — Enforcement Priority Maps</h1>
  <ul>
    <li><a href="morning_09h.html">Morning Peak — 09:00</a></li>
    <li><a href="evening_17h.html">Evening Peak — 17:00</a></li>
    <li><a href="day_schedule.html">Full Day Schedule</a></li>
  </ul>
</body>
</html>
```

Copy each HTML output to `docs/` with a descriptive filename, then push.

---

## Troubleshooting

| Problem | Fix |
|:---|:---|
| 404 on first load | Wait 2–3 minutes and hard-refresh. First deploy takes time. |
| Map renders blank / no tiles | The Folium map uses OpenStreetMap tiles which require an internet connection. Works in browsers with internet access. |
| "Page not found" after push | Confirm that `docs/index.html` exists and is committed (run `git status` to check) |
| Settings → Pages not visible | You may need to make the repository public, or you need admin access to enable Pages on private repos |
| Old map still showing | GitHub Pages CDN caches aggressively. Add `?v=2` to the URL to bust cache: `https://...github.io/GridLock-R2/?v=2` |

---

## Summary

```
docs/index.html  ←─ copy of enforcement_priority_<date>_<hour>h.html
     │
     ├── git add + git commit + git push origin main
     │
     └── GitHub Pages (Settings → Pages → main branch → /docs folder)
              │
              └── Public URL: https://<username>.github.io/GridLock-R2/
```

The demo map is now permanently accessible via a public URL you can paste into a judge's browser or include in a slide deck.
