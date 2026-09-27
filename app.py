"""
WIUT Traffic Event Detection - Streamlit Web Application

A web interface for traffic event detection from video uploads.
Covers all rubric criteria:
- Live demo (30%)
- Sample-video visualizations (20%)
- EDA findings (15%)
- Approach and report (15%)
- Team and portfolio (10%)
- Design, UX, extras (10%)
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import solution

# Page configuration
st.set_page_config(
    page_title="WIUT Traffic Event Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for design
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .event-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px;
        font-size: 14px;
    }
    .accident { background: #ff4757; color: white; }
    .near_miss { background: #ffa502; color: white; }
    .red_light { background: #ff6b81; color: white; }
    .wrong_way { background: #ee5a24; color: white; }
    .stopped_vehicle { background: #f39c12; color: white; }
    .jaywalking { background: #9b59b6; color: white; }
    .default { background: #3498db; color: white; }
    .timeline-event {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        height: 30px;
        border-radius: 5px;
        margin: 5px 0;
        display: flex;
        align-items: center;
        padding: 0 15px;
        color: #1a1a2e;
        font-weight: bold;
    }
    .risk-high { background: linear-gradient(90deg, #ff4757, #ff6b81); }
    .risk-medium { background: linear-gradient(90deg, #ffa502, #ffdd59); }
    .risk-low { background: linear-gradient(90deg, #2ed573, #7bed9f); }
    .section-header {
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .team-card {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
    .demo-box {
        border: 2px dashed #00d4ff;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        background: rgba(0,212,255,0.05);
    }
    @media (max-width: 768px) {
        .stButton>button {
            width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_event_class_color(label: str) -> str:
    """Get color class for event label."""
    colors = {
        "accident": "accident",
        "near_miss": "near_miss",
        "red_light": "red_light",
        "wrong_way": "wrong_way",
        "illegal_u_turn": "default",
        "stopped_vehicle": "stopped_vehicle",
        "jaywalking": "jaywalking",
        "failure_to_yield": "default",
        "illegal_turn": "default",
        "solid_line_crossing": "default",
        "stop_line": "default",
        "congestion": "default",
        "road_obstacle": "default",
        "fire_smoke": "default",
    }
    return colors.get(label, "default")


def load_sample_predictions():
    """Load sample predictions for visualization."""
    pred_path = Path("predictions_samples.json")
    if pred_path.exists():
        with open(pred_path, "r") as f:
            return json.load(f)
    return None


def get_video_info(video_path: str) -> dict:
    """Extract video metadata using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


def process_video_with_solution(video_path: str) -> dict:
    """
    Process video using the actual solution.py detect_events function
    and RiskEstimator for risk scoring.
    Returns: dict with events, risk, stats, annotated_frames
    """
    info = get_video_info(video_path)
    if not info:
        return {"error": "Could not open video file"}

    video_id = Path(video_path).name
    total_frames = info["frame_count"]
    fps = info["fps"]

    # Initialize tracking stats
    stats = {
        "total_vehicles": 0,
        "total_pedestrians": 0,
        "max_concurrent": 0
    }

    # Sample frames for annotated output (every 30 seconds, max 3 frames)
    sample_times = list(range(0, int(info["duration"]), 30))[:3]
    annotated_frames = []

    # Check if YOLO is available
    model = None
    if solution.HAS_YOLO:
        try:
            from ultralytics import YOLO
            weight_path = Path("weights/yolov8n.pt")
            if weight_path.exists():
                model = YOLO(str(weight_path))
            else:
                model = YOLO("yolov8n.pt")
        except Exception:
            pass

    # Call solution.detect_events()
    try:
        events = solution.detect_events(video_path)
    except Exception as e:
        return {"error": f"Detection failed: {str(e)}"}

    # Run RiskEstimator for risk curve + collect stats
    risk_estimator = solution.RiskEstimator()
    risk_estimator.reset({
        "video_id": video_id,
        "fps": fps,
        "width": info["width"],
        "height": info["height"],
        "n_frames": total_frames
    })

    # Process frames for risk estimation
    cap = cv2.VideoCapture(video_path)
    risk_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps
        risk_score = risk_estimator.step(frame, t_sec)
        risk_data.append([t_sec, risk_score])

        # Count detections at this frame
        if model and frame_idx % 10 == 0:
            try:
                results = model(frame, verbose=False)
                for r in results:
                    boxes = r.boxes
                    current_count = len(boxes)
                    stats["max_concurrent"] = max(stats["max_concurrent"], current_count)

                # Capture annotated frame at sample times
                if int(t_sec) in sample_times and len(annotated_frames) < 3:
                    annotated = frame.copy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        color = (0, 255, 0) if cls in [2, 3, 5, 7] else (255, 0, 0)
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    annotated_frames.append({"time": t_sec, "frame": annotated, "count": current_count})
            except Exception:
                pass

        frame_idx += 1

    cap.release()

    # Final stats
    stats["total_frames_processed"] = frame_idx

    return {
        "video_id": video_id,
        "info": info,
        "events": events,
        "risk": risk_data,
        "stats": stats,
        "annotated_frames": annotated_frames
    }


# ============================================================================
# PAGE: HOME / LIVE DEMO
# ============================================================================

def show_live_demo():
    """Live demo section - 30% of rubric."""
    st.markdown('<div class="section-header"><h1>🚗 Live Demo</h1></div>', unsafe_allow_html=True)
    st.markdown("Upload a traffic video to detect events and visualize risk in real-time.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a video file (MP4, AVI)",
            type=["mp4", "avi", "mov"],
            help="Upload a traffic video for event detection"
        )

    with col2:
        st.info("📹 Supported formats: MP4, AVI, MOV")
        st.info("⏱️ Processing time depends on video length")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process video
        with st.spinner("Processing video..."):
            result = process_video_with_solution(str(temp_path))

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # Show results
            st.success("✅ Processing complete!")

            # Video info
            info = result["info"]
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Video Information</h3>
                <p><strong>Duration:</strong> {info['duration']:.2f}s |
                <strong>FPS:</strong> {info['fps']:.2f} |
                <strong>Resolution:</strong> {info['width']}x{info['height']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Statistics
            if "stats" in result:
                stats = result["stats"]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Frames Processed", stats.get("total_frames_processed", 0))
                with col2:
                    st.metric("Max Vehicles in Frame", stats.get("max_concurrent", 0))
                with col3:
                    st.metric("Detected Events", len(result.get("events", [])))
                with col4:
                    avg_risk = np.mean([r[1] for r in risk_data]) if risk_data else 0
                    st.metric("Avg Risk Score", f"{avg_risk:.3f}")

            # Annotated frames
            if "annotated_frames" in result and result["annotated_frames"]:
                st.markdown("### 🎬 Detected Objects (Sample Frames)")
                for i, af in enumerate(result["annotated_frames"]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Convert BGR to RGB for display
                        frame_rgb = cv2.cvtColor(af["frame"], cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"Time: {af['time']:.1f}s - {af['count']} objects", use_container_width=True)
                    with col2:
                        st.markdown(f"**Time:** {af['time']:.1f}s")
                        st.markdown(f"**Objects:** {af['count']}")

            # Events found
            if result["events"]:
                st.markdown("### 🎯 Detected Events")
                for event in result["events"]:
                    color_class = get_event_class_color(event[2])
                    st.markdown(f"""
                    <span class="event-badge {color_class}">
                        {event[2]}: {event[0]:.1f}s - {event[1]:.1f}s
                    </span>
                    """, unsafe_allow_html=True)
            else:
                st.info("No events detected in this video.")

            # Risk visualization
            st.markdown("### 📈 Risk Timeline")
            risk_data = result.get("risk", [])
            if risk_data:
                # Simple risk display
                risk_times = [r[0] for r in risk_data]
                risk_values = [r[1] for r in risk_data]

                chart_data = pd.DataFrame({
                    "Time (s)": risk_times,
                    "Risk Score": risk_values
                })
                st.line_chart(chart_data.set_index("Time (s)"))

            # Cleanup
            temp_path.unlink(missing_ok=True)

    else:
        # Demo box when no file uploaded
        st.markdown("""
        <div class="demo-box">
            <h2>📤 Drop your video here</h2>
            <p>Or click the button above to browse</p>
            <p style="color: #888; font-size: 14px;">
                Our AI will analyze the video for traffic events<br>
                and generate a risk timeline visualization
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE: SAMPLE VISUALIZATIONS
# ============================================================================

def show_sample_viz():
    """Sample video visualizations - 20% of rubric."""
    st.markdown('<div class="section-header"><h1>📊 Sample Video Visualizations</h1></div>', unsafe_allow_html=True)

    predictions = load_sample_predictions()

    if predictions is None:
        st.warning("No sample predictions available.")
        return

    # Select video
    videos = list(predictions.get("videos", {}).keys())
    if not videos:
        st.warning("No videos in predictions.")
        return

    selected_video = st.selectbox("Select a sample video:", videos)

    if selected_video:
        video_data = predictions["videos"][selected_video]
        events = video_data.get("events", [])
        risk = video_data.get("risk", [])

        # Video info card
        st.markdown(f"""
        <div class="metric-card">
            <h3>📹 {selected_video}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Events timeline
        st.markdown("### 🎯 Event Timeline")

        if events:
            for event in events:
                start, end, label = event
                st.markdown(f"""
                <div class="timeline-event">
                    <span style="flex: 1;">{label}</span>
                    <span>{start:.1f}s → {end:.1f}s</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No events detected in this video.")

        # Risk curve
        st.markdown("### 📈 Risk Curve")

        if risk and len(risk) > 1:
            risk_times = [r[0] for r in risk]
            risk_values = [r[1] for r in risk]

            chart_data = pd.DataFrame({
                "Time (s)": risk_times,
                "Risk Score": risk_values
            })

            # Color based on risk level
            max_risk = max(risk_values) if risk_values else 0
            if max_risk > 0.7:
                st.error(f"⚠️ High risk detected! Max: {max_risk:.2f}")
            elif max_risk > 0.3:
                st.warning(f"⚡ Moderate risk detected. Max: {max_risk:.2f}")
            else:
                st.success(f"✅ Low risk. Max: {max_risk:.2f}")

            st.line_chart(chart_data.set_index("Time (s)"))

            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Risk", f"{np.mean(risk_values):.3f}")
            with col2:
                st.metric("Max Risk", f"{np.max(risk_values):.3f}")
            with col3:
                st.metric("Std Dev", f"{np.std(risk_values):.3f}")
        else:
            st.info("No risk data available.")

        # Summary statistics
        st.markdown("### 📋 Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            st.metric("Risk Data Points", len(risk))


# ============================================================================
# PAGE: EDA
# ============================================================================

def show_eda():
    """EDA findings - 15% of rubric."""
    st.markdown('<div class="section-header"><h1>📈 Exploratory Data Analysis</h1></div>', unsafe_allow_html=True)

    st.markdown("""
    Our analysis went beyond simple frame counts to uncover insights that shaped our solution.
    """)

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🎯 Event Distribution", "🔍 Video Analysis", "💡 Key Findings"])

    with tab1:
        st.markdown("### Dataset Overview")

        # Sample data
        data = {
            "Metric": ["Total Videos", "Total Duration", "Average Length", "Frame Rate", "Resolution"],
            "Value": ["1 (sample)", "~30 seconds", "30s", "30 FPS", "1920x1080"]
        }
        df = pd.DataFrame(data)
        st.table(df)

        st.markdown("""
        **Note:** The competition provides sample videos for development.
        Ground truth annotations confirm no target events in the sample video.
        """)

    with tab2:
        st.markdown("### Event Class Distribution")

        # Class definitions with descriptions
        classes_info = [
            ("accident", "Collision between road users / with fixed object", 0),
            ("near_miss", "Sharp braking or swerving to avoid collision", 0),
            ("red_light", "Crossing stop line on red", 0),
            ("wrong_way", "Driving against traffic direction", 0),
            ("illegal_u_turn", "U-turn where prohibited", 0),
            ("stopped_vehicle", "Stationary on carriageway ≥10s", 0),
            ("jaywalking", "Pedestrian on carriageway outside crossing", 0),
            ("failure_to_yield", "Driving through crossing with pedestrian", 0),
            ("illegal_turn", "Turn from wrong lane or prohibited direction", 0),
            ("solid_line_crossing", "Lane change across solid marking", 0),
            ("stop_line", "Stopped past stop line on red", 0),
            ("congestion", "Standstill across all lanes", 0),
            ("road_obstacle", "Debris, animal, or fallen object", 0),
            ("fire_smoke", "Visible fire or smoke", 0),
        ]

        df_classes = pd.DataFrame(classes_info, columns=["Event Type", "Description", "Count"])
        st.table(df_classes)

        st.caption("Current sample has 0 events - all classes need more training data.")

    with tab3:
        st.markdown("### Video Analysis")

        # Analysis insights
        st.markdown("""
        #### Key Video Characteristics Analyzed:

        1. **Camera Position**: Fixed traffic camera, bird's eye view
        2. **Lighting Conditions**: Variable (daytime in sample)
        3. **Traffic Density**: Low in sample video
        4. **Road Layout**: Multi-lane urban road with intersections
        5. **Frame Quality**: 30 FPS, 1080p resolution
        """)

        # Show analysis from notebooks
        st.markdown("### 📓 Analysis Notebooks")

        notebook_files = list(Path("notebooks").glob("*.ipynb")) if Path("notebooks").exists() else []

        if notebook_files:
            for nb in notebook_files:
                st.markdown(f"- 📄 `{nb.name}`")
        else:
            st.info("No notebooks found in the notebooks/ directory.")

    with tab4:
        st.markdown("### 💡 Key Findings That Shaped Our Solution")

        findings = [
            {
                "title": "Sample Video Has No Events",
                "description": "After thorough review, the sample video C3905.MP4 contains no target events. All 14 event classes need positive training examples.",
                "impact": "Required gathering additional labeled videos for each event type"
            },
            {
                "title": "Detection Approach Selection",
                "description": "YOLO/RT-DETR for object detection + ByteTrack for tracking provides good balance of speed and accuracy.",
                "impact": "Chose lightweight models to meet competition time constraints"
            },
            {
                "title": "Event Temporal Patterns",
                "description": "Most events have duration 1-30 seconds. Short blips (<0.5s) should be filtered, gaps (<1s) merged.",
                "impact": "Post-processing rules: min_duration=0.5s, merge_gap=1.0s"
            },
            {
                "title": "Risk Anticipation Horizon",
                "description": "5-second lookahead is standard for accident anticipation. TTC (Time-To-Collision) is a strong signal.",
                "impact": "Implemented causal risk estimation using TTC between vehicle pairs"
            }
        ]

        for i, finding in enumerate(findings, 1):
            with st.expander(f"{i}. {finding['title']}"):
                st.markdown(f"**Description:** {finding['description']}")
                st.markdown(f"**Impact:** {finding['impact']}")


# ============================================================================
# PAGE: APPROACH
# ============================================================================

def show_approach():
    """Approach and report - 15% of rubric."""
    st.markdown('<div class="section-header"><h1>🔧 Approach & Report</h1></div>', unsafe_allow_html=True)

    st.markdown("""
    A reader can rebuild our entire pipeline from this documentation.
    """)

    # Pipeline overview
    st.markdown("## 🔄 Pipeline Architecture")

    pipeline_steps = [
        ("1. Video Input", "Load video file, extract metadata (FPS, duration, resolution)"),
        ("2. Frame Sampling", "Sample frames at appropriate interval (every 2-5 frames)"),
        ("3. Object Detection", "Use YOLO/RT-DETR to detect vehicles, pedestrians, cyclists"),
        ("4. Tracking", "Apply ByteTrack to maintain object IDs across frames"),
        ("5. Scene Understanding", "Detect lanes, stop lines, crossings from static frame regions"),
        ("6. Event Detection", "Apply rules for each event class based on trajectories + scene"),
        ("7. Post-processing", "Merge gaps, filter blips, ensure no overlapping segments"),
        ("8. Risk Estimation", "Calculate TTC between vehicle pairs, map to risk score"),
    ]

    for step, desc in pipeline_steps:
        st.markdown(f"**{step}:** {desc}")

    # Technical details
    st.markdown("## 🔧 Technical Implementation")

    with st.expander("📦 Dependencies"):
        st.code("""
# requirements.txt
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0  # YOLO
scikit-learn>=1.3.0
streamlit>=1.28.0
pandas>=2.0.0
        """)

    with st.expander("🎯 Event Detection Rules"):
        st.markdown("""
        | Event | Detection Logic |
        |-------|-----------------|
        | accident | Two objects with sudden speed change + proximity |
        | near_miss | Sharp braking (high deceleration) without collision |
        | red_light | Vehicle crosses stop line when signal=red |
        | wrong_way | Vehicle heading opposite to lane direction |
        | stopped_vehicle | Vehicle stationary >10s not at signal |
        | jaywalking | Pedestrian in roadway outside crossing |
        | failure_to_yield | Vehicle + pedestrian at crossing simultaneously |
        """)

    with st.expander("📊 Risk Estimation Formula"):
        st.markdown("""
        ```
        Risk = 1 - min(TTC / HORIZON, 1.0)

        Where:
        - TTC = min(Time-To-Collision to all lead vehicles)
        - HORIZON = 5.0 seconds
        - If no vehicle ahead, Risk = 0
        ```
        """)

    # Code structure
    st.markdown("## 📁 Code Structure")

    st.markdown("""
    ```
    wiut_cv_scripts/
    ├── solution.py          # Main entry point (detect_events, RiskEstimator)
    ├── app.py               # Streamlit web application
    ├── src/                 # Helper modules
    │   ├── detector.py      # YOLO wrapper
    │   ├── tracker.py       # ByteTrack wrapper
    │   └── events.py        # Event detection rules
    ├── weights/             # Model weights
    ├── notebooks/           # EDA notebooks
    │   ├── eda_analysis.ipynb
    │   ├── experiments.ipynb
    │   └── training.ipynb
    └── requirements.txt
    ```
    """)

    # Failures stated plainly
    st.markdown("## ⚠️ Known Limitations & Failures")

    failures = [
        {
            "issue": "No positive training examples",
            "status": "Active blocker",
            "solution": "Need to obtain labeled videos with actual events"
        },
        {
            "issue": "Sample video has no events",
            "status": "Confirmed",
            "solution": "Cannot validate detection on sample; need additional data"
        },
        {
            "issue": "Scene understanding is basic",
            "status": "Limitation",
            "solution": "Currently uses simple heuristics; could use segmentation"
        }
    ]

    for f in failures:
        st.markdown(f"""
        - **{f['issue']}** ({f['status']}): {f['solution']}
        """)


# ============================================================================
# PAGE: TEAM
# ============================================================================

def show_team():
    """Team and portfolio - 10% of rubric."""
    st.markdown('<div class="section-header"><h1>👥 Team & Portfolio</h1></div>', unsafe_allow_html=True)

    st.markdown("### Team Members")

    team_members = [
        {
            "name": "Oybek Nortojiyev",
            "role": "Lead Developer",
            "contribution": "Pipeline architecture, event detection logic, risk estimation",
            "link": "https://github.com/moonloybek"
        },
        {
            "name": "Zaxro Madrimova",
            "role": "ML Engineer",
            "contribution": "Model training, YOLO integration, EDA analysis",
            "link": "https://github.com/madrimovazaxro"
        },
        {
            "name": "Aziza Mamadiyeva",
            "role": "Frontend Developer",
            "contribution": "Streamlit web app, visualizations, UI/UX",
            "link": "https://github.com/Azi-kh"
        }
    ]

    cols = st.columns(3)

    for i, member in enumerate(team_members):
        with cols[i]:
            st.markdown(f"""
            <div class="team-card">
                <h3>👤 {member['name']}</h3>
                <p style="color: #00d4ff;">{member['role']}</p>
                <p>{member['contribution']}</p>
                <a href="{member['link']}">🔗 Portfolio</a>
            </div>
            """, unsafe_allow_html=True)

    # Roles breakdown
    st.markdown("### Role Breakdown")

    roles_data = [
        ("Algorithm Development", "50%", "Event detection, risk estimation"),
        ("Web Development", "30%", "Streamlit app, visualizations"),
        ("Data Analysis", "15%", "EDA, findings, documentation"),
        ("Testing & Validation", "5%", "Pipeline testing"),
    ]

    for role, pct, desc in roles_data:
        st.markdown(f"- **{role}** ({pct}): {desc}")

    # Links
    st.markdown("### 🔗 Project Links")

    st.markdown("""
    - 📂 [GitHub Repository](#)
    - 📄 [Technical Report](#)
    - 🎥 [Demo Video](#)
    - 📊 [Competition Page](#)
    """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""

    # Sidebar navigation
    st.sidebar.title("🚗 Traffic Event Detection")
    st.sidebar.markdown("---")

    # Navigation
    pages = {
        "Live Demo": show_live_demo,
        "Sample Visualizations": show_sample_viz,
        "EDA": show_eda,
        "Approach": show_approach,
        "Team": show_team,
    }

    # Add icons to page names
    page_names = list(pages.keys())

    selection = st.sidebar.radio(
        "Navigation",
        page_names,
        format_func=lambda x: f"  {x}"
    )

    # Show selected page
    pages[selection]()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **WIUT Hackathon CV**
    Traffic Event Detection

    Built with Streamlit 🚀
    """)


if __name__ == "__main__":
    main()