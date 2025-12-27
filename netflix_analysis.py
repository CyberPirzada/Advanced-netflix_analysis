import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'netflix_titles.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Datafile not found: {path}")
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine='openpyxl')
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # director
    if 'director' in df.columns:
        df['director'] = df['director'].fillna('Unknown')

    # date_added -> added_year, added_month
    if 'date_added' in df.columns:
        df['date_added_parsed'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['added_year'] = df['date_added_parsed'].dt.year
        df['added_month'] = df['date_added_parsed'].dt.month
    else:
        df['date_added_parsed'] = pd.NaT
        df['added_year'] = pd.NA
        df['added_month'] = pd.NA

    # duration: parse minutes or seasons
    def parse_duration(x):
        if pd.isna(x):
            return np.nan, 0
        s = str(x).strip()
        if 'min' in s:
            try:
                return int(s.split()[0]), 0
            except Exception:
                return np.nan, 0
        if 'Season' in s or 'Seasons' in s:
            try:
                return np.nan, int(s.split()[0])
            except Exception:
                return np.nan, 0
        return np.nan, 0

    durations = df['duration'].apply(lambda x: parse_duration(x) if 'duration' in df.columns else (np.nan, 0))
    df['duration_minutes'] = [d[0] for d in durations]
    df['seasons'] = [d[1] for d in durations]

    # genres
    if 'listed_in' in df.columns:
        df['genres'] = df['listed_in'].fillna('').apply(lambda s: [g.strip() for g in s.split(',') if g.strip()])
    else:
        df['genres'] = [[] for _ in range(len(df))]

    # director popularity
    df['director_popularity'] = df['director'].map(df['director'].value_counts())

    return df


def save_plot(fig_path):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)


def exploratory_plots(df: pd.DataFrame, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)

    # Titles over years
    if 'added_year' in df.columns:
        yearly = df.dropna(subset=['added_year']).groupby(['added_year', 'type']).size().unstack(fill_value=0)
        plt.figure(figsize=(10, 5))
        yearly.plot(kind='line', marker='o')
        plt.title('Titles added per year by Type')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.tight_layout()
        fn = os.path.join(out_dir, 'titles_per_year.png')
        plt.savefig(fn)
        plt.close()

    # Top genres
    all_genres = Counter(g for sub in df['genres'] for g in sub)
    top_genres = all_genres.most_common(10)
    if top_genres:
        names, counts = zip(*top_genres)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(counts), y=list(names), palette='viridis')
        plt.title('Top 10 Genres')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'top_genres.png'))
        plt.close()

    # Type distribution
    plt.figure(figsize=(6, 4))
    df['type'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Type Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'type_distribution.png'))
    plt.close()


def basic_ml(df: pd.DataFrame, out_dir='plots'):
    # Task: predict `type` (Movie vs TV Show) using simple features
    data = df.copy()
    data = data[['type', 'duration_minutes', 'seasons', 'added_year', 'director_popularity', 'genres']].copy()
    data = data.dropna(subset=['type'])

    # target: encode Movie=0, TV Show=1
    data['target'] = (data['type'] != 'Movie').astype(int)

    # feature: top-6 genres one-hot
    all_genres = Counter(g for sub in df['genres'] for g in sub)
    top_genres = [g for g, _ in all_genres.most_common(6)]
    for g in top_genres:
        data[f'genre_{g}'] = data['genres'].apply(lambda gl: int(g in gl))

    X = data[['duration_minutes', 'seasons', 'added_year', 'director_popularity'] + [f'genre_{g}' for g in top_genres]].fillna(0)
    y = data['target']

    if len(X) < 50:
        return {'note': 'Not enough data for ML'}, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    # save a simple importance plot
    importances = clf.feature_importances_
    feat_names = X.columns.tolist()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances, y=feat_names)
    plt.title('Feature importances (RandomForest)')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'feature_importances.png'))
    plt.close()

    return {'accuracy': acc, 'report': report}, clf


def write_report(df: pd.DataFrame, ml_results: dict, out_path='netflix_analysis_report.md'):
    lines = []
    lines.append('# Netflix Analysis Report')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat()}')
    lines.append('')
    lines.append('## Dataset summary')
    lines.append(f'- Rows: {len(df):,}')
    lines.append(f'- Columns: {len(df.columns)}')
    lines.append('')
    # top genres
    all_genres = Counter(g for sub in df['genres'] for g in sub)
    lines.append('## Top genres')
    for g, c in all_genres.most_common(10):
        lines.append(f'- {g}: {c}')
    lines.append('')

    lines.append('## ML Summary')
    if ml_results is None or 'note' in ml_results:
        lines.append('- Not enough data to run the ML experiment')
    else:
        lines.append(f"- Accuracy: {ml_results['accuracy']:.3f}")
        # short classification breakdown
        rep = ml_results['report']
        lines.append('- Classification Report (summary):')
        for k, v in rep.items():
            if k in ['0', '1']:
                lines.append(f"  - Class {k}: precision={v['precision']:.2f}, recall={v['recall']:.2f}, f1={v['f1-score']:.2f}")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main(args):
    df = load_data(args.file)
    df = clean_dataframe(df)

    out_dir = args.out_dir
    exploratory_plots(df, out_dir=out_dir)
    ml_results, model = basic_ml(df, out_dir=out_dir)
    # ml_results may be dict or (dict, model)
    if isinstance(ml_results, tuple):
        ml_results = ml_results[0]
    write_report(df, ml_results, out_path=os.path.join(out_dir, '..', 'netflix_analysis_report.md'))
    print('Plots and report saved to', os.path.abspath(out_dir))
    if ml_results and 'accuracy' in ml_results:
        print('ML accuracy:', ml_results['accuracy'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Netflix dataset exploratory analysis and basic ML')
    parser.add_argument('--file', help='Path to netflix csv/xlsx', default=None)
    parser.add_argument('--out-dir', help='Directory to save plots and report', default='plots')
    args = parser.parse_args()
    main(args)
