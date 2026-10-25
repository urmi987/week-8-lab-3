"""One frozen classifier, two annotation-defined environments; run from any directory."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
IDENTITIES = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white']
SEED = 735


def metrics(y, p):
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean())
              for b in range(10) if np.any(bins == b))
    return dict(n=len(y), prevalence=float(y.mean()), accuracy=accuracy_score(y, pred),
                balanced_accuracy=balanced_accuracy_score(y, pred),
                auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
                toxic_recall=tp/(tp+fn) if tp+fn else np.nan,
                false_positive_rate=fp/(fp+tn) if fp+tn else np.nan,
                brier=brier_score_loss(y, p), ece=ece,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def run():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    path = ROOT / 'data/all_data_with_identities.csv'
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert digest == protocol['data_sha256'], 'Dataset checksum mismatch'
    df = pd.read_csv(path)
    original_n = len(df)
    original_splits = df['split'].value_counts().to_dict()
    required = ['comment_text', 'toxicity', 'muslim']
    missing = df[required].isna().any(axis=1) | df['comment_text'].fillna('').str.strip().eq('')
    df = df.loc[~missing].copy()
    # Normalize only for duplicate detection, never use annotations as predictors.
    df['_key'] = df.comment_text.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    # Keep training before validation before test when duplicate text crosses splits.
    df['_priority'] = df['split'].map({'train': 0, 'val': 1, 'test': 2})
    assert df['_priority'].notna().all()
    df = df.sort_values('_priority', kind='stable')
    duplicates = int(df.duplicated('_key').sum())
    df = df.drop_duplicates('_key').copy()
    df['y'] = (df.toxicity >= 0.5).astype(int)
    df['environment'] = np.where(df.muslim >= 0.5, 'Shifted', 'ID')
    train = df.loc[(df['split'] == 'train') & (df.environment == 'ID')].copy()
    id_test = df.loc[(df['split'] == 'test') & (df.environment == 'ID')].copy()
    shifted = df.loc[(df['split'] == 'test') & (df.environment == 'Shifted')].copy()
    assert train.muslim.lt(.5).all() and id_test.muslim.lt(.5).all()
    assert shifted.muslim.ge(.5).all()
    sets = [set(x['_key']) for x in [train, id_test, shifted]]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=3, sublinear_tf=True),
        LogisticRegression(C=1, solver='liblinear', max_iter=1000, random_state=SEED))
    print(f'Fitting once on {len(train):,} development comments', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train.comment_text, train.y)
    rows, subgroup_rows, group_label_rows, calibration_rows = [], [], [], []
    tests = {'ID': id_test, 'Shifted': shifted}
    for env, frame in tests.items():
        frame['p'] = model.predict_proba(frame.comment_text)[:, 1]
        rows.append({'environment': env, **metrics(frame.y.to_numpy(), frame.p.to_numpy())})
        for identity in IDENTITIES:
            part = frame.loc[frame[identity] >= .5]
            if len(part):
                subgroup_rows.append({'environment': env, 'identity': identity,
                                      **metrics(part.y.to_numpy(), part.p.to_numpy())})
                for label in [0, 1]:
                    cell = part.loc[part.y == label]
                    if len(cell):
                        group_label_rows.append(dict(environment=env, identity=identity, label=label,
                            n=len(cell), accuracy=float(((cell.p >= .5) == cell.y).mean())))
        for b in range(10):
            cell = frame.loc[np.minimum((frame.p * 10).astype(int), 9) == b]
            if len(cell):
                calibration_rows.append(dict(environment=env, bin=b, n=len(cell),
                    mean_probability=float(cell.p.mean()), observed_toxicity=float(cell.y.mean())))
    performance = pd.DataFrame(rows).set_index('environment')
    performance.to_csv(out / 'performance.csv')
    subgroups = pd.DataFrame(subgroup_rows)
    subgroups.to_csv(out / 'subgroups.csv', index=False)
    group_labels = pd.DataFrame(group_label_rows)
    group_labels.to_csv(out / 'identity_label_accuracy.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(out / 'calibration.csv', index=False)
    # IID row bootstrap: binary accuracy can be resampled exactly with binomial draws.
    rng = np.random.default_rng(SEED)
    boot = {env: rng.binomial(int(performance.loc[env, 'n']), performance.loc[env, 'accuracy'], 1000)
                  / performance.loc[env, 'n'] for env in tests}
    delta_ci = np.quantile(boot['Shifted'] - boot['ID'], [.025, .975]).tolist()
    environment_rows = []
    for name, frame in {'Development': train, **tests}.items():
        row = dict(environment=name, n=len(frame), prevalence=float(frame.y.mean()),
                   mean_characters=float(frame.comment_text.str.len().mean()))
        row.update({i: float((frame[i] >= .5).mean()) for i in IDENTITIES})
        environment_rows.append(row)
    pd.DataFrame(environment_rows).to_csv(out / 'environments.csv', index=False)
    # Largest shifts in TF-IDF coordinates, standardized using development variation.
    vectorizer = model[0]
    matrices = {name: vectorizer.transform(frame.comment_text) for name, frame in
                {'Development': train, **tests}.items()}
    means = {name: np.asarray(x.mean(axis=0)).ravel() for name, x in matrices.items()}
    sd = np.sqrt(np.maximum(np.asarray(matrices['Development'].power(2).mean(axis=0)).ravel()
                            - means['Development']**2, 0))
    smd = (means['Shifted'] - means['ID']) / np.maximum(sd, 1e-8)
    pd.DataFrame(dict(feature=vectorizer.get_feature_names_out(), id_mean=means['ID'],
                     shifted_mean=means['Shifted'], standardized_difference=smd))\
      .assign(absolute_difference=lambda d: d.standardized_difference.abs())\
      .sort_values('absolute_difference', ascending=False).head(15).to_csv(out / 'feature_shift.csv', index=False)
    oov = {}
    analyzer = vectorizer.build_analyzer()
    vocab = vectorizer.vocabulary_
    for env, frame in tests.items():
        counts = [(len(t), sum(x not in vocab for x in t)) for t in map(analyzer, frame.comment_text)]
        oov[env] = sum(x[1] for x in counts) / sum(x[0] for x in counts)
    split_ids = pd.concat([x[['id', 'split', 'environment']].assign(role=role)
                           for role, x in {'Development': train, **tests}.items()])
    split_ids.to_csv(out / 'split_manifest.csv', index=False)
    versions = {m: importlib.metadata.version(m) for m in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'jupyter', 'tabulate']}
    provenance = dict(data_sha256=digest, original_n=original_n, original_splits=original_splits,
        missing_removed=int(missing.sum()), duplicate_text_removed=duplicates,
        accuracy_delta_ci=delta_ci, seed=SEED, oov_ngram_rate=oov,
        python=platform.python_version(), platform=platform.platform(), packages=versions,
        template_commit=protocol['starter_commit'],
        model_iterations=model[1].n_iter_.tolist(),
        checks=['checksum verified', 'required fields complete', 'environment membership verified',
                'no normalized text overlap between train and either test or between tests',
                'single fit', 'training-only vocabulary', 'convergence warning treated as error'])
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(performance.to_string(), flush=True)
    print('Accuracy delta 95% interval:', delta_ci, flush=True)
    return provenance


if __name__ == '__main__':
    run()
