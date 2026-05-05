import re
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime
import base64
from PIL import Image
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# =========================================================
# CONFIG
# =========================================================

DATA_PATH = Path("telegram_vacancies_cleaned.csv")
MODEL_PATH = Path("telegram_role_classifier_svm.joblib")
CONTACT_PATH = Path("contact_messages.csv")
ASSETS_DIR = Path("assets")

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

SKILL_KEYWORDS = [
    "python", "sql", "excel", "power bi", "tableau", "pandas", "numpy",
    "machine learning", "data analysis", "data visualization", "statistics",
    "analytics", "dashboard", "dashboards", "etl", "airflow", "spark", "dbt",
    "bigquery",

    "javascript", "typescript", "react", "vue", "angular", "html", "css",
    "next.js", "node.js", "express",

    "java", "spring", "php", "laravel", "go", "golang", "c#", ".net",
    "fastapi", "django", "flask", "rest api", "graphql",

    "postgresql", "mysql", "mongodb", "redis", "kafka", "elasticsearch",

    "docker", "kubernetes", "linux", "git", "ci/cd", "jenkins",
    "terraform", "aws", "azure", "gcp",

    "selenium", "playwright", "cypress", "postman", "testing", "qa",
    "manual testing", "automation testing",

    "figma", "ui", "ux", "prototyping", "wireframe",

    "jira", "scrum", "agile", "project management", "product management",

    "cybersecurity", "security", "siem", "soc", "network security",
    "penetration testing", "owasp",

    "database", "dba", "backup", "replication", "query optimization",
]


SUPPORTED_ROLES = [
    "Data Analyst",
    "Data Scientist",
    "Backend Developer",
    "Frontend Developer",
    "QA Engineer",
    "DevOps Engineer",
    "UI/UX Designer",
    "Project Manager",
    "Cybersecurity Analyst",
    "Database Administrator",
]


# =========================================================
# STREAMLIT PAGE SETTINGS
# =========================================================

logo_icon = Image.open("assets/logo.png")

st.set_page_config(
    page_title="SkillMatch Jobs",
    page_icon=logo_icon,
    layout="wide",
)

# =========================================================
# STYLES
# =========================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        padding: 58px 46px;
        border-radius: 30px;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 52%, #db2777 100%);
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 22px 50px rgba(37, 99, 235, 0.25);
    }

    .hero h1 {
        font-size: 58px;
        font-weight: 950;
        margin-bottom: 12px;
        line-height: 1.04;
    }

    .hero p {
        font-size: 20px;
        opacity: 0.96;
        max-width: 920px;
        line-height: 1.6;
    }

    .card {
        background: #ffffff;
        padding: 25px;
        border-radius: 24px;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }

    .small-card {
        background: #ffffff;
        padding: 22px;
        border-radius: 22px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
        border: 1px solid #e5e7eb;
        height: 100%;
        margin-bottom: 16px;
    }

    .section-title {
        font-size: 32px;
        font-weight: 900;
        color: #111827;
        margin-top: 16px;
        margin-bottom: 14px;
    }

    .muted {
        color: #64748b;
        font-size: 15px;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-weight: 700;
        font-size: 13px;
        margin: 4px 5px 4px 0;
        border: 1px solid #c7d2fe;
    }

    .skill-badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
        font-size: 13px;
        margin: 4px 5px 4px 0;
        border: 1px solid #bbf7d0;
    }

    .missing-badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: #ffedd5;
        color: #9a3412;
        font-weight: 700;
        font-size: 13px;
        margin: 4px 5px 4px 0;
        border: 1px solid #fed7aa;
    }

    .vacancy-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 24px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    .vacancy-title {
        font-size: 22px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 7px;
    }

    .role-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        font-weight: 800;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .score {
        font-size: 30px;
        font-weight: 950;
        color: #2563eb;
    }

    .image-card {
        background: white;
        padding: 14px;
        border-radius: 24px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }

    .placeholder-photo {
        min-height: 265px;
        border-radius: 24px;
        background: linear-gradient(135deg, #dbeafe, #ede9fe, #fce7f3);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #334155;
        font-size: 22px;
        font-weight: 900;
        text-align: center;
        padding: 24px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.6);
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 28px 0 12px 0;
    }

    div.stButton > button {
        border-radius: 14px;
        font-weight: 800;
        border: none;
        padding: 0.75rem 1rem;
    }

    .stTextArea textarea {
        border-radius: 16px;
    }

    .stTextInput input {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"@[\\w_]+", " ", text)
    text = re.sub(r"#(\\w+)", r" \\1 ", text)
    text = text.replace("\\n", " ").replace("/", " ")
    text = re.sub(r"[^a-zа-яё0-9+#.\\s-]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> list[str]:
    text = clean_text(text)
    found = []

    for skill in SKILL_KEYWORDS:
        pattern = r"(?<!\\w)" + re.escape(skill.lower()) + r"(?!\\w)"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(list(set(found)))


def safe_get(row, column: str, default=""):
    if column in row.index and pd.notna(row[column]):
        return row[column]
    return default


def render_badges(items, css_class="badge"):
    if not items:
        st.markdown("<span class='muted'>No items found.</span>", unsafe_allow_html=True)
        return

    html = ""
    for item in items:
        item = str(item).strip()
        if item:
            html += f"<span class='{css_class}'>{item}</span>"
    st.markdown(html, unsafe_allow_html=True)


def show_local_image(image_name: str, placeholder_text: str = "Image"):
    image_path = ASSETS_DIR / image_name

    if image_path.exists():
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.image(str(image_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div class="placeholder-photo">
                {placeholder_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


def parse_skills_from_cell(value) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).lower().strip()

    for ch in ["[", "]", "'", '"', "{", "}"]:
        text = text.replace(ch, "")

    parts = re.split(r"[,;|]+", text)
    skills = [p.strip() for p in parts if p.strip()]

    if not skills:
        skills = extract_skills(text)

    return sorted(list(set(skills)))


def get_text_preview(text, limit=460) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def collect_top_skills_from_vacancies(similar_df, top_n=12):
    all_skills = []

    for _, row in similar_df.iterrows():
        if "skills_extracted" in similar_df.columns:
            all_skills.extend(parse_skills_from_cell(row["skills_extracted"]))
        else:
            all_skills.extend(extract_skills(row["text_for_model"]))

    all_skills = [skill.lower().strip() for skill in all_skills if skill.strip()]
    return Counter(all_skills).most_common(top_n)


def find_first_existing_column(df, possible_columns):
    for col in possible_columns:
        if col in df.columns:
            return col
    return None


# =========================================================
# DATA + MODEL FUNCTIONS
# =========================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame(), "File telegram_vacancies_cleaned.csv not found."

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        return pd.DataFrame(), f"Could not read CSV file: {e}"

    required_columns = ["text_for_model", "role_guess"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        return pd.DataFrame(), f"Missing required columns: {missing}"

    df = df.dropna(subset=["text_for_model", "role_guess"]).copy()

    if "job_title" not in df.columns:
        df["job_title"] = df["role_guess"].astype(str) + " Vacancy"

    if "skills_extracted" not in df.columns:
        df["skills_extracted"] = df["text_for_model"].apply(
            lambda x: ", ".join(extract_skills(x))
        )

    if "skill_count" not in df.columns:
        df["skill_count"] = df["skills_extracted"].apply(
            lambda x: len(parse_skills_from_cell(x))
        )

    df["skill_count"] = pd.to_numeric(df["skill_count"], errors="coerce").fillna(0)

    return df, ""


@st.cache_resource
def load_or_train_model(df):
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            MODEL_PATH.unlink(missing_ok=True)

    train_df = df[df["role_guess"].astype(str).str.lower() != "other"].copy()

    if train_df.empty:
        train_df = df.copy()

    if train_df["role_guess"].nunique() < 2:
        st.error("Model training requires at least 2 different roles in role_guess column.")
        st.stop()

    min_df_value = 2 if len(train_df) >= 30 else 1

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=min_df_value,
                max_df=0.95
            )
        ),
        (
            "clf",
            LinearSVC(
                class_weight="balanced",
                random_state=42
            )
        ),
    ])

    model.fit(train_df["text_for_model"], train_df["role_guess"])
    joblib.dump(model, MODEL_PATH)

    return model


@st.cache_resource
def build_search_index(df):
    min_df_value = 2 if len(df) >= 30 else 1

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=min_df_value,
        max_df=0.95
    )

    matrix = vectorizer.fit_transform(df["text_for_model"])
    return vectorizer, matrix


def predict_top_roles(model, text, top_n=3):
    cleaned = clean_text(text)
    predicted_role = model.predict([cleaned])[0]

    try:
        clf = model.named_steps["clf"]
        classes = list(clf.classes_)
        scores = model.decision_function([cleaned])

        if len(classes) == 2:
            single_score = float(scores[0])
            scores_list = [-single_score, single_score]
        else:
            scores_list = list(scores[0])

        role_scores = list(zip(classes, scores_list))
        role_scores = sorted(role_scores, key=lambda x: x[1], reverse=True)[:top_n]

        raw_scores = [score for _, score in role_scores]
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        results = []
        for role, score in role_scores:
            if max_score == min_score:
                percentage = 85
            else:
                percentage = int(62 + ((score - min_score) / (max_score - min_score)) * 33)

            results.append({
                "role": str(role),
                "score": int(max(1, min(99, percentage)))
            })

        return str(predicted_role), results

    except Exception:
        return str(predicted_role), [{"role": str(predicted_role), "score": 85}]


def find_similar_vacancies(text, df, vectorizer, matrix, top_n=10):
    query = vectorizer.transform([clean_text(text)])
    scores = cosine_similarity(query, matrix).ravel()
    best_indexes = scores.argsort()[::-1][:top_n]

    result = df.iloc[best_indexes].copy()
    result["similarity"] = scores[best_indexes]
    return result


def calculate_missing_skills(user_skills, similar_df, top_n=10):
    vacancy_skills = []

    for _, row in similar_df.iterrows():
        if "skills_extracted" in similar_df.columns:
            vacancy_skills.extend(parse_skills_from_cell(row["skills_extracted"]))
        else:
            vacancy_skills.extend(extract_skills(row["text_for_model"]))

    vacancy_skills = [s.lower().strip() for s in vacancy_skills if s.strip()]
    user_set = set([s.lower().strip() for s in user_skills])

    common = Counter(vacancy_skills).most_common(30)

    missing = []
    for skill, _ in common:
        if skill not in user_set and skill not in missing:
            missing.append(skill)

    return missing[:top_n]


def save_contact_message(name, email, message):
    file_exists = CONTACT_PATH.exists()

    with open(CONTACT_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["created_at", "name", "email", "message"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            email,
            message,
        ])


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

df, data_error = load_data()

if df.empty:
    st.error(data_error or "Could not load dataset.")
    st.info("Put telegram_vacancies_cleaned.csv in the same folder as streamlit_app.py.")
    st.stop()

model = load_or_train_model(df)
search_vectorizer, search_matrix = build_search_index(df)


# =========================================================
# SIDEBAR
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

pages = [
    "Home",
    "Find Vacancy",
    "Vacancy Results",
    "Dataset Overview",
    "Contact",
]

st.sidebar.markdown("##  SkillMatch Jobs")
st.sidebar.markdown("AI vacancy matching from real dataset")

selected_page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(st.session_state.page),
)

st.session_state.page = selected_page

st.sidebar.divider()
st.sidebar.markdown("### Dataset")
st.sidebar.metric("Rows", len(df))
st.sidebar.metric("Roles", df["role_guess"].nunique())

clear_rows_sidebar = len(df[df["role_guess"].astype(str).str.lower() != "other"])
st.sidebar.metric("Clear Rows", clear_rows_sidebar)


# =========================================================
# PAGE: HOME
# =========================================================

def page_home():
    st.markdown(
        """
        <div class="hero">
            <h1>SkillMatch Jobs</h1>
            <p>
                Find suitable IT vacancies based on your skills.
                Write your skills, tools, and experience — the system will analyze
                real Telegram vacancy data and recommend matching vacancies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_col, text_col = st.columns([0.9, 1.1])

    with image_col:
        show_local_image("image.png", "SkillMatch Jobs")

    with text_col:
        st.markdown(
            """
            <div class="card">
                <h2>AI-powered vacancy matching</h2>
                <p class="muted">
                    This website does not show random vacancies. It analyzes your skills
                    and finds the most similar real vacancies from the uploaded Telegram vacancy dataset.
                </p>
                <p class="muted">
                    The system uses machine learning role prediction and text similarity search.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cta_col, _ = st.columns([0.35, 0.65])
        with cta_col:
            if st.button("Find My Vacancy", type="primary", use_container_width=True):
                st.session_state.page = "Find Vacancy"
                st.rerun()

    st.markdown("<div class='section-title'>Why This Website?</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="small-card">
                <h3> Skill-Based Matching</h3>
                <p class="muted">
                    The app analyzes your skills and compares them with real vacancy requirements.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="small-card">
                <h3> Real Dataset</h3>
                <p class="muted">
                    Vacancy recommendations are selected from telegram_vacancies_cleaned.csv.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="small-card">
                <h3>🤖 AI Prediction</h3>
                <p class="muted">
                    TF-IDF and Linear SVM predict your most suitable IT role.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Supported Roles</div>", unsafe_allow_html=True)
    render_badges(SUPPORTED_ROLES, "badge")

    st.markdown("<div class='section-title'>Dataset Snapshot</div>", unsafe_allow_html=True)

    
    m1, m2, m3, m4 = st.columns(4)

    total_vacancies = len(df)
    unique_roles = df["role_guess"].nunique()
    clear_rows = len(df[df["role_guess"].astype(str).str.lower() != "other"])

    all_skills = []
    for value in df["skills_extracted"].dropna():
        all_skills.extend(parse_skills_from_cell(value))

    top_skill = Counter(all_skills).most_common(1)[0][0] if all_skills else "N/A"

    m1.metric("Total Vacancies", total_vacancies)
    m2.metric("Unique Roles", unique_roles)
    m3.metric("Clear Rows", clear_rows)
    m4.metric("Top Skill", top_skill.title())

    
    st.markdown("<div class='section-title'>How It Works</div>", unsafe_allow_html=True)

    h1, h2, h3, h4 = st.columns(4)

    steps = [
        ("1", "Enter Skills", "Write your skills, tools, and preferred tasks."),
        ("2", "AI Analysis", "The model predicts your suitable role."),
        ("3", "Vacancy Match", "The system finds similar real vacancies."),
        ("4", "Improve Skills", "You get missing skills and recommendations."),
    ]

    for col, step in zip([h1, h2, h3, h4], steps):
        with col:
            st.markdown(
                f"""
                <div class="small-card">
                    <h2>{step[0]}</h2>
                    <h4>{step[1]}</h4>
                    <p class="muted">{step[2]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# PAGE: FIND VACANCY
# =========================================================

def page_find_vacancy():
    st.markdown("<div class='section-title'>Find Suitable Vacancy</div>", unsafe_allow_html=True)

    intro_col, img_col = st.columns([1.15, 0.85])

    with intro_col:
        st.markdown(
            """
            <div class="card">
                <h3>Write your profile</h3>
                <p class="muted">
                    Enter your skills, tools, experience level, and preferred tasks.
                    The app will predict your role and find similar vacancies from the real dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with img_col:
        show_local_image("image1.png", "AI Vacancy Matching")

    left, right = st.columns([1.2, 0.8])

    with left:
        with st.form("vacancy_form"):
            skills = st.text_area(
                "Your Skills",
                placeholder="Python, SQL, pandas, Excel, Power BI, dashboards",
                height=130,
            )

            tools = st.text_area(
                "Tools",
                placeholder="PostgreSQL, Tableau, Jupyter Notebook, Docker",
                height=100,
            )

            c1, c2 = st.columns(2)

            with c1:
                experience_level = st.selectbox(
                    "Experience Level",
                    ["Entry", "Junior", "Middle", "Senior"],
                )

            with c2:
                job_type = st.selectbox(
                    "Preferred Job Type",
                    ["Any", "Remote", "Office", "Hybrid"],
                )

            tasks = st.text_area(
                "Preferred Tasks",
                placeholder="data cleaning, dashboard building, API development, testing, DevOps automation",
                height=120,
            )

            submit = st.form_submit_button(
                "Show Suitable Vacancies",
                type="primary",
                use_container_width=True,
            )

        if submit:
            if not skills or len(skills.strip()) < 5:
                st.warning("Please write at least a few skills.")
                return

            user_text = f"""
            Skills: {skills}
            Tools: {tools}
            Experience level: {experience_level}
            Preferred job type: {job_type}
            Preferred tasks: {tasks}
            """

            predicted_role, top_roles = predict_top_roles(model, user_text)
            detected_skills = extract_skills(user_text)

            similar = find_similar_vacancies(
                user_text,
                df,
                search_vectorizer,
                search_matrix,
                top_n=10,
            )

            missing_skills = calculate_missing_skills(detected_skills, similar)

            st.session_state["result"] = {
                "user_text": user_text,
                "predicted_role": predicted_role,
                "top_roles": top_roles,
                "detected_skills": detected_skills,
                "similar": similar,
                "missing_skills": missing_skills,
            }

            st.success("Done! Opening Vacancy Results page.")
            st.session_state.page = "Vacancy Results"
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="small-card">
                <h3>Good Example</h3>
                <p class="muted">
                    Python, SQL, pandas, Excel, Power BI, dashboards,
                    data cleaning, business analysis, statistics.
                </p>
                <hr>
                <h3>Tip</h3>
                <p class="muted">
                    Write programming languages, tools, frameworks,
                    preferred tasks, and experience level.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Popular Skills")
        render_badges([
            "Python", "SQL", "React", "Docker", "Power BI",
            "PostgreSQL", "Testing", "Figma", "FastAPI"
        ], "badge")


# =========================================================
# PAGE: VACANCY RESULTS
# =========================================================

def page_results():
    st.markdown("<div class='section-title'>Vacancy Results</div>", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.info("No results yet. Go to Find Vacancy page and enter your skills.")
        if st.button("Go to Find Vacancy", type="primary"):
            st.session_state.page = "Find Vacancy"
            st.rerun()
        return

    result = st.session_state["result"]

    predicted_role = result["predicted_role"]
    top_roles = result["top_roles"]
    detected_skills = result["detected_skills"]
    similar = result["similar"]
    missing_skills = result["missing_skills"]

    st.markdown(
        f"""
        <div class="card">
            <p class="muted">Best Matching Role</p>
            <h1>{predicted_role}</h1>
            <p>
                Your skills match this role because similar vacancy texts contain
                related skills, tools, and job requirements.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Top Predicted Roles")

    role_cols = st.columns(min(len(top_roles), 3))

    for col, item in zip(role_cols, top_roles):
        with col:
            st.markdown(
                f"""
                <div class="small-card">
                    <h3>{item["role"]}</h3>
                    <div class="score">{item["score"]}%</div>
                    <p class="muted">Estimated role match</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(item["score"] / 100)

    st.markdown("### Detected Skills")
    render_badges(detected_skills, "skill-badge")

    st.markdown("### Recommended Skills to Improve")
    render_badges(missing_skills, "missing-badge")

    st.divider()
    st.markdown("### Visual Analysis")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if top_roles:
            top_roles_df = pd.DataFrame(top_roles)

            fig_roles = px.bar(
                top_roles_df,
                x="role",
                y="score",
                title="Top Predicted Roles",
                text="score",
            )
            fig_roles.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_roles.update_layout(
                yaxis_title="Match Percentage",
                xaxis_title="Role",
                yaxis_range=[0, 105],
            )
            st.plotly_chart(fig_roles, use_container_width=True)

    with chart_col2:
        if not similar.empty:
            sim_df = similar.head(8).copy()
            sim_df["similarity_percent"] = (sim_df["similarity"] * 100).round(1)
            sim_df["short_title"] = sim_df["job_title"].astype(str).str.slice(0, 35)

            fig_sim = px.bar(
                sim_df,
                x="similarity_percent",
                y="short_title",
                orientation="h",
                title="Most Similar Real Vacancies",
                text="similarity_percent",
            )
            fig_sim.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_sim.update_layout(
                xaxis_title="Similarity Percentage",
                yaxis_title="Vacancy",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_sim, use_container_width=True)

    skill_counts = collect_top_skills_from_vacancies(similar, top_n=12)

    if skill_counts:
        skill_df = pd.DataFrame(skill_counts, columns=["skill", "count"])

        fig_skills = px.bar(
            skill_df,
            x="count",
            y="skill",
            orientation="h",
            title="Most Common Skills in Matching Vacancies",
        )
        fig_skills.update_layout(
            xaxis_title="Frequency",
            yaxis_title="Skill",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_skills, use_container_width=True)

    st.divider()

    st.markdown("### Similar Real Vacancies")
    st.info("These vacancies are selected from the uploaded Telegram vacancy dataset, not randomly generated.")

    link_column = find_first_existing_column(
        df,
        ["link", "url", "source_url", "telegram_link", "post_link"]
    )

    company_column = find_first_existing_column(
        df,
        ["company", "company_name", "employer"]
    )

    location_column = find_first_existing_column(
        df,
        ["location", "city", "work_location"]
    )

    salary_column = find_first_existing_column(
        df,
        ["salary", "salary_range", "compensation"]
    )

    source_column = find_first_existing_column(
        df,
        ["source", "channel", "telegram_channel"]
    )

    for i, (_, row) in enumerate(similar.iterrows(), start=1):
        title = safe_get(row, "job_title", "Untitled Vacancy")
        role = safe_get(row, "role_guess", "Unknown Role")
        skills_text = safe_get(row, "skills_extracted", "")
        skill_count = safe_get(row, "skill_count", "")
        vacancy_text = safe_get(row, "text_for_model", "")
        similarity = float(safe_get(row, "similarity", 0))
        similarity_percent = round(similarity * 100, 1)

        st.markdown(
            f"""
            <div class="vacancy-card">
                <div class="vacancy-title">{i}. {title}</div>
                <span class="role-pill">{role}</span>
                <p><b>Similarity:</b> {similarity_percent}%</p>
                <p><b>Skill count:</b> {skill_count}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if company_column:
            company = safe_get(row, company_column, "")
            if company:
                st.write(f"**Company:** {company}")

        if location_column:
            location = safe_get(row, location_column, "")
            if location:
                st.write(f"**Location:** {location}")

        if salary_column:
            salary = safe_get(row, salary_column, "")
            if salary:
                st.write(f"**Salary:** {salary}")

        if source_column:
            source = safe_get(row, source_column, "")
            if source:
                st.write(f"**Source:** {source}")

        if link_column:
            link = safe_get(row, link_column, "")
            if link and str(link).startswith(("http://", "https://")):
                st.link_button("Open Source", str(link))

        vacancy_skills = parse_skills_from_cell(skills_text)
        if vacancy_skills:
            st.markdown("**Vacancy Skills:**")
            render_badges(vacancy_skills[:18], "badge")

        st.write("**Preview:**")
        st.write(get_text_preview(vacancy_text, 520))

        with st.expander("Show full vacancy text"):
            st.write(vacancy_text)

        st.divider()

    st.markdown("### Learning Roadmap")

    r1, r2, r3, r4 = st.columns(4)

    roadmap = [
        ("1", "Learn Missing Skills", "Focus on skills repeated in similar vacancies."),
        ("2", "Build Portfolio", "Create one project related to your predicted role."),
        ("3", "Prepare CV", "Add skills, tools, and project results clearly."),
        ("4", "Apply", "Apply to vacancies similar to your profile."),
    ]

    for col, item in zip([r1, r2, r3, r4], roadmap):
        with col:
            st.markdown(
                f"""
                <div class="small-card">
                    <h2>{item[0]}</h2>
                    <h4>{item[1]}</h4>
                    <p class="muted">{item[2]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# PAGE: DATASET OVERVIEW
# =========================================================

def page_dataset_overview():
    st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)

    top_col, image_col = st.columns([1.15, 0.85])

    with top_col:
        total_rows = len(df)
        clear_rows = len(df[df["role_guess"].astype(str).str.lower() != "other"])
        roles_count = df["role_guess"].nunique()
        avg_skill_count = round(df["skill_count"].mean(), 2)

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Total Rows", total_rows)
        m2.metric("Clear Rows", clear_rows)
        m3.metric("Unique Roles", roles_count)
        m4.metric("Avg Skill Count", avg_skill_count)

    with image_col:
        show_local_image("data_dashboard.png", "Dataset Dashboard")

    st.divider()

    col1, col2 = st.columns(2)

    role_counts = df["role_guess"].value_counts().reset_index()
    role_counts.columns = ["role", "count"]

    with col1:
        st.markdown("### Role Distribution")
        fig = px.bar(
            role_counts,
            x="role",
            y="count",
            title="Vacancies by Role",
        )
        fig.update_layout(xaxis_title="Role", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Role Share")
        fig = px.pie(
            role_counts,
            names="role",
            values="count",
            title="Role Share",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Skills")

    all_skills = []
    if "skills_extracted" in df.columns:
        for value in df["skills_extracted"].dropna():
            all_skills.extend(parse_skills_from_cell(value))

    if all_skills:
        skill_counts = Counter(all_skills).most_common(20)
        skill_df = pd.DataFrame(skill_counts, columns=["skill", "count"])

        fig = px.bar(
            skill_df,
            x="count",
            y="skill",
            orientation="h",
            title="Top 20 Skills in Dataset",
        )
        fig.update_layout(
            xaxis_title="Count",
            yaxis_title="Skill",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No skills_extracted column found.")

    st.markdown("### Skill Count Distribution")

    fig_hist = px.histogram(
        df,
        x="skill_count",
        nbins=20,
        title="Distribution of Skill Count per Vacancy",
    )
    fig_hist.update_layout(
        xaxis_title="Number of Skills",
        yaxis_title="Number of Vacancies",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Average Skill Count by Role")

    avg_skill_by_role = (
        df.groupby("role_guess")["skill_count"]
        .mean()
        .reset_index()
        .sort_values("skill_count", ascending=False)
    )

    fig_avg = px.bar(
        avg_skill_by_role,
        x="role_guess",
        y="skill_count",
        title="Average Skill Count by Role",
    )
    fig_avg.update_layout(
        xaxis_title="Role",
        yaxis_title="Average Skill Count",
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    source_col = find_first_existing_column(
        df,
        ["source_type", "source", "channel", "telegram_channel"]
    )

    if source_col:
        st.markdown("### Source Distribution")
        source_counts = df[source_col].fillna("Unknown").value_counts().reset_index()
        source_counts.columns = ["source", "count"]

        fig_source = px.pie(
            source_counts,
            names="source",
            values="count",
            title="Source Distribution",
        )
        st.plotly_chart(fig_source, use_container_width=True)

    st.markdown("### Browse Dataset Examples")

    selected_role = st.selectbox(
        "Choose role",
        df["role_guess"].value_counts().index.tolist(),
    )

    examples = df[df["role_guess"] == selected_role].head(12)

    visible_cols = []
    for col in ["job_title", "role_guess", "skills_extracted", "skill_count"]:
        if col in examples.columns:
            visible_cols.append(col)

    st.dataframe(
        examples[visible_cols],
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# PAGE: CONTACT
# =========================================================

def page_contact():
    st.markdown("<div class='section-title'>Contact Us</div>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown(
            """
            <div class="card">
                <h3>Send a Message</h3>
                <p class="muted">
                    Have questions or want to improve this project?
                    Send a message below. It will be saved to contact_messages.csv.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("contact_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message", height=160)

            submitted = st.form_submit_button(
                "Submit Message",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if not name.strip() or not email.strip() or not message.strip():
                st.warning("Please fill in all fields.")
            else:
                save_contact_message(name, email, message)
                st.success("Your message has been saved successfully.")

    with right:
        show_local_image("5463079743910516210.jpg", "Contact SkillMatch Jobs")

        st.markdown(
            """
            <div class="small-card">
                <h3>Project Info</h3>
                <p><b>Project:</b> SkillMatch Jobs</p>
                <p><b>Email:</b> example@email.com</p>
                <p><b>Location:</b> Kazakhstan</p>
                <p class="muted">
                    This project recommends vacancies using a Telegram vacancy dataset
                    and machine learning similarity search.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# ROUTER
# =========================================================

if st.session_state.page == "Home":
    page_home()

elif st.session_state.page == "Find Vacancy":
    page_find_vacancy()

elif st.session_state.page == "Vacancy Results":
    page_results()

elif st.session_state.page == "Dataset Overview":
    page_dataset_overview()

elif st.session_state.page == "Contact":
    page_contact()


st.markdown(
    """
    <div class="footer">
        SkillMatch Jobs © 2026 | AI-based vacancy matching from real Telegram vacancy dataset
    </div>
    """,
    unsafe_allow_html=True,
)
