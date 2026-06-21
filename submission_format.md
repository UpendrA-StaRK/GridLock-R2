# Create Submission
## Prototype Round 2

### Title *
*Give your submission a clear, descriptive title.*

GridLock R2: AI-Driven Parking Intelligence for Bengaluru Traffic Police

### Description *
*Describe your project, solution, or idea. You can use formatting and links.*

**Problem Statement Direction:**
> "How can AI-driven parking intelligence detect illegal parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?"

**The Problem:** Bengaluru generates tens of thousands of illegal parking violations per day, but enforcement is currently reactive and based on patrol memory. Traffic officers start their day with a critical question: *where do I go first?*

**The Solution:** **GridLock R2** directly answers the hackathon's operational challenge. It is a predictive AI system that ingests historical police violation records to identify true congestion-driving parking clusters. Using a LightGBM predictive model (trained on 150 days of real Bengaluru violation data) and geospatial clustering (DBSCAN), GridLock generates a prioritized, hour-by-hour enforcement schedule. It tells officers exactly which zones to prioritize at what time of day, ensuring every patrol car maximizes congestion reduction per kilometer driven. 

**Real-World Impact:** Unlike a static frequency lookup table, our model adapts dynamically to the time of day, capturing late-night hotspots vs morning rush-hour peaks. On our held-out test set, the model achieves a **38.2% improvement in count prediction** over the naive baseline and consistently beats it in per-hour ranking accuracy (NDCG). To make this actionable, we calculate a **Congestion Impact Score (CIS)**—multiplying the AI's predicted violation density by a junction-presence weight—to rank zones based on the actual traffic disruption they cause. 

### Theme *
*Select the theme that best fits your submission.*

Urban Mobility / Smart City / Traffic Management

### Snapshots
*Upload images of your project (JPG, JPEG or PNG, up to 3MB each).*

[Please manually upload screenshots of the `docs/index.html` interactive map and `data/outputs/shap_summary.png` here]

### Video URL *
*Share a link to your product demo or pitch video.*

[Please replace this with your demo video URL]

### Presentation *
*Upload your pitch deck or slides (.key, .odp, .odt, .pdf, .pps, .ppt, .pptx). Max 50MB.*

[Please manually upload your presentation file here]

### Demo Link *
*Share the link to your working demo or prototype.*

[Please replace this with your GitHub Pages URL or Live Demo link]

### Repository URL *
*Share the link to your code repository (e.g. GitHub, Bitbucket).*

[Please replace this with your GitHub Repository URL]

### Source Code *
*Upload your source code (e.g. zip or apk). Max 50MB.*

[Please upload the generated `source_code.zip` here]

### Instructions to Run *
*Add steps to run your project so reviewers can test it.*

**1. Automated Fresh System Setup (Train from Scratch)**
To install dependencies, set up the environment, and run the entire pipeline from scratch (which automatically trains a new model):
- **Windows**: Run `run_project.bat`
- **Linux/Mac**: Run `./run_project.sh`

**2. Quick Run (Inference Only)**
If you want to view the dashboard instantly without retraining:
1. Ensure the pre-trained checkpoint files (`model.lgb`, `features.yaml`, etc.) are placed inside a folder named `submission_checkpoint/` at the root of the project.
2. Run the inference-only scripts:
   - **Windows**: Run `run_inference_only.bat`
   - **Linux/Mac**: Run `./run_inference_only.sh`
3. Open `docs/index.html` in your browser to view the interactive 24-hour enforcement map.

### Custom Attachment
*Upload any additional file (e.g. .zip, .7z, .rar)*

[Optional: You can upload your pre-trained model checkpoint zip here if applicable]
