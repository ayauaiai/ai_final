# SkillMatch Jobs

## Public Website Link

**Project Name:** SkillMatch Jobs  
**Repository:** https://github.com/ayauaiai/ai_final 

**Live Website:** https://aifinal-yzwkfktoddvbqlbyus9uzw.streamlit.app/

## Project Overview

**SkillMatch Jobs** is a Streamlit web application that helps users find suitable IT vacancies based on their skills, tools, experience level, and preferred tasks.

The user enters their profile information, and the system predicts the most suitable job role using machine learning. After that, it finds similar vacancies from a real Telegram vacancy dataset.

The project does **not** use random vacancy data. All vacancy recommendations are selected from the uploaded dataset using text similarity.

---

## Main Goal

The main goal of this project is to help job seekers, students, and beginner IT specialists understand which IT vacancies match their current skills.

The application can be used to:

- Find a suitable IT job role
- Analyze user skills
- Discover similar real vacancies
- Understand missing skills
- Explore vacancy dataset statistics
- Support career direction decisions

---

## Key Features

- Modern Streamlit website interface
- Skill-based vacancy matching
- Machine learning role prediction
- Real vacancy recommendations from dataset
- Similar vacancy search using cosine similarity
- Detected skills analysis
- Missing skills recommendation
- Top predicted roles
- Vacancy result cards
- Dataset dashboard with charts
- Contact form
- Simple and responsive layout

---

## Technologies Used

The project was built using:

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Plotly
- Pillow

---

## Machine Learning Approach

The system uses a machine learning pipeline for role prediction.

### Model Pipeline


TF-IDF Vectorizer
+
Linear SVM / LinearSVC Classifier


### Main ML Steps

1. Load cleaned vacancy dataset
2. Use vacancy text as input data
3. Train a KNN + LinearSVC model
4. Predict the most suitable job role
5. Use cosine similarity to find similar vacancies
6. Display the most relevant vacancy results to the user

---

## Similarity Search

The app uses **cosine similarity** to compare the user's input with vacancy texts from the dataset.

User input includes:

- Skills
- Tools
- Experience level
- Preferred job type
- Preferred tasks

The system then returns the most similar vacancies from the dataset.

---

## Dataset

The project uses the following dataset file:

```text
telegram_vacancies_cleaned.csv
```

The dataset contains cleaned Telegram vacancy data.

### Important Dataset Columns

```text
job_title
role_guess
text_for_model
skills_extracted
skill_count
```

### Column Descriptions

| Column | Description |
|---|---|
| `job_title` | Vacancy title |
| `role_guess` | Predicted or labeled job role |
| `text_for_model` | Cleaned text used for ML model and similarity search |
| `skills_extracted` | Skills extracted from vacancy text |
| `skill_count` | Number of detected skills in the vacancy |

---

## Website Pages

### 1. Home

The Home page introduces the project and explains how the system works.

It includes:

- Hero section
- Project description
- Main features
- Supported roles
- Dataset snapshot
- Simple explanation of the process

---

### 2. Find Vacancy

This page contains the main input form.

The user enters:

- Skills
- Tools
- Experience level
- Preferred job type
- Preferred tasks

After submitting the form, the app predicts the best matching role and searches for similar vacancies.

---

### 3. Vacancy Results

This page shows the recommendation results.

It includes:

- Best matching role
- Top predicted roles
- Detected user skills
- Recommended skills to improve
- Similar real vacancies from the dataset
- Vacancy similarity scores
- Vacancy text preview
- Full vacancy text
- Visual analysis charts
- Learning roadmap

---

### 4. Dataset Overview

This page works as a dataset dashboard.

It shows:

- Total rows
- Clear rows
- Unique roles
- Average skill count
- Role distribution chart
- Role share chart
- Top skills chart
- Skill count distribution
- Average skill count by role
- Dataset examples

---

### 5. Contact

The Contact page allows users to send a message.

The form includes:

- Name
- Email
- Message

Submitted messages are saved locally into:

```text
contact_messages.csv
```

---

## Supported Roles

The system supports different IT-related roles, such as:

- Data Analyst
- Data Scientist
- Backend Developer
- Frontend Developer
- QA Engineer
- DevOps Engineer
- UI/UX Designer
- Project Manager
- Cybersecurity Analyst
- Database Administrator

---

## Project Structure

```text
ai_final/
│
├── streamlit_app.py
├── requirements.txt
├── telegram_vacancies_cleaned.csv
│
├── assets/
│   ├── logo.png
│   ├── image.png
│   ├── image1.png
│   ├── data_dashboard.png
│   └── 5463079743910516210.jpg
│
└── README.md
```

---

## Installation and Local Run

### 1. Clone the repository

```bash
git clone https://github.com/ayauaiai/ai_final.git
cd ai_final
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

or:

```bash
python -m streamlit run streamlit_app.py
```

---

## Requirements

The `requirements.txt` file should include:

```text
streamlit
pandas
scikit-learn
joblib
plotly
pillow
```

---

## Deployment

The project can be deployed using **Streamlit Community Cloud**.

### Deployment Settings

```text
Repository: ayauaiai/ai_final
Branch: main
Main file path: streamlit_app.py
```

### Deployment Steps

1. Push the project to GitHub
2. Open Streamlit Community Cloud
3. Connect GitHub account
4. Choose repository
5. Select branch `main`
6. Set main file path as `streamlit_app.py`
7. Click Deploy
8. Copy the public Streamlit app link

---

## Example User Input

```text
Skills:
Python, SQL, pandas, Excel, Power BI, dashboards

Tools:
PostgreSQL, Tableau, Jupyter Notebook

Experience level:
Junior

Preferred job type:
Remote

Preferred tasks:
Data cleaning, dashboard building, business analysis, reporting
```

---

## Example Output

The app may return:

```text
Best Matching Role: Data Analyst

Detected Skills:
Python, SQL, pandas, Excel, Power BI

Recommended Skills to Improve:
Statistics, ETL, Tableau, Machine Learning

Similar Vacancies:
Real vacancy results selected from telegram_vacancies_cleaned.csv
```

---

## Disclaimer

This project uses a prepared Telegram vacancy dataset.

The app does not scrape live job websites and does not guarantee that the displayed vacancies are currently available.

Vacancy recommendations are based on dataset similarity and machine learning prediction.

---

## Author

Created as an AI final project.

**Project Name:** SkillMatch Jobs  
**Repository:** https://github.com/ayauaiai/ai_final  
**Live Website:** https://aifinal-yzwkfktoddvbqlbyus9uzw.streamlit.app/
