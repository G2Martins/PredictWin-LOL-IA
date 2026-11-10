# Treina o modelo exatamente como main.py e exporta tudo que o site mostra (o navegador só desenha).
# Rodar da raiz do repositório:  python -I export_model.py
# Saídas: site/src/app/data/{model,eda,fixtures}.json e os downloads em site/public/data/
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

root = Path(__file__).resolve().parent
out = root / "site" / "src" / "app" / "data"
pub = root / "site" / "public" / "data"
out.mkdir(parents=True, exist_ok=True)
pub.mkdir(parents=True, exist_ok=True)

SLOTS = ['team_1__001', 'team_1__002', 'team_1__003', 'team_1__004', 'team_1__005',
         'team_2__001', 'team_2__002', 'team_2__003', 'team_2__004', 'team_2__005']
# Ordem das colunas no s11.csv segue as rotas (confirmado pelos campeões mais frequentes de cada coluna).
ROLES = ['Topo', 'Selva', 'Meio', 'Atirador', 'Suporte']


def convert_game_length_to_minutes(game_length):
    # Igual a main.py.
    try:
        minutes, seconds = game_length.split('m')
        return int(minutes.strip()) + int(seconds.replace('s', '').strip()) / 60
    except:  # noqa: E722
        return 0


# ---------- Treino: idêntico a main.py ----------
data = pd.read_csv(root / 'assets' / 's11.csv')
data['game_length_minutes'] = data['game_length'].apply(convert_game_length_to_minutes)
X = data[SLOTS + ['game_length_minutes']]
y = data['result'].apply(lambda x: 1 if x == 'Victory' else 0)

encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[SLOTS])
X_encoded = sp.hstack([X_encoded, X[['game_length_minutes']].values])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
idx_train, idx_test = train_test_split(np.arange(len(data)), test_size=0.3, random_state=42)
assert (y.iloc[idx_test].values == y_test.values).all()  # mesmas linhas do split original

model = LogisticRegression(max_iter=5000, C=1, class_weight='balanced')
model.fit(X_train, y_train)
print(f"Acurácia do modelo: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")


# ---------- Exportação ----------
def r4(v):
    return round(float(v), 4)  # só para gráficos; coeficientes vão com precisão total


def metrics(Xs, ys):
    pred = model.predict(Xs)
    proba = model.predict_proba(Xs)[:, 1]
    rep = classification_report(ys, pred, output_dict=True, zero_division=0)
    per = {name: {k: rep[str(c)][k] for k in ('precision', 'recall', 'f1-score', 'support')}
           for c, name in [(0, 'Derrota'), (1, 'Vitória')]}
    return {
        'n': int(len(ys)),
        'accuracy': accuracy_score(ys, pred),
        'balancedAccuracy': balanced_accuracy_score(ys, pred),
        'rocAuc': roc_auc_score(ys, proba),
        'majorityBaseline': float(max(ys.mean(), 1 - ys.mean())),
        'perClass': per,
        'confusion': confusion_matrix(ys, pred, labels=[0, 1]).tolist(),
    }


coef = model.coef_[0]
offset = 0
slots = []
for i, (key, cats) in enumerate(zip(SLOTS, encoder.categories_)):
    counts = data[key].value_counts()
    slots.append({
        'key': key,
        'team': 1 if i < 5 else 2,
        'role': ROLES[i % 5],
        'coef': {c: float(coef[offset + j]) for j, c in enumerate(cats)},
        'count': {c: int(counts.get(c, 0)) for c in cats},
    })
    offset += len(cats)
assert offset == len(coef) - 1

minutes = data['game_length_minutes']
ts = pd.to_datetime(data['timestamp'])
Xcsr = X_encoded.tocsr()
proba_all = model.predict_proba(Xcsr)[:, 1]
p_test = proba_all[idx_test]


def row_input(i):
    r = data.iloc[i]
    return {'team1': [r[s] for s in SLOTS[:5]], 'team2': [r[s] for s in SLOTS[5:]],
            'minutes': float(r['game_length_minutes'])}


# ---------- Visualizações do modelo ----------
def roc(ys, ps, n=80):
    fpr, tpr, _ = roc_curve(ys, ps)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(n, len(fpr))).round().astype(int))
    return [[r4(fpr[k]), r4(tpr[k])] for k in keep]


prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=10, strategy='uniform')
counts, _ = np.histogram(p_test, bins=np.linspace(0, 1, 11))
calib = [{'pred': r4(a), 'obs': r4(b), 'n': int(n)} for a, b, n in zip(prob_pred, prob_true, counts[counts > 0])]

# Modelo linear: p depende só de S (soma dos pesos dos 10 campeões) + coef x minutos, então o plano (S, minutos) é exato.
beta, b0 = float(coef[-1]), float(model.intercept_[0])
S = model.decision_function(Xcsr) - b0 - beta * minutes.values
s_grid = np.linspace(np.floor(S[idx_test].min()), np.ceil(S[idx_test].max()), 21)
m_grid = np.linspace(8, 50, 8)
s_mid, m_mid = s_grid[:-1] + np.diff(s_grid) / 2, m_grid[:-1] + np.diff(m_grid) / 2
grid = [[r4(1 / (1 + np.exp(-(b0 + si + beta * mi)))) for si in s_mid] for mi in m_mid]
pts = np.random.default_rng(0).choice(idx_test, size=200, replace=False)

model_json = {
    'source': 'assets/s11.csv (repositório original)',
    'params': {'algorithm': 'LogisticRegression', 'solver': model.solver, 'C': 1, 'classWeight': 'balanced',
               'maxIter': 5000, 'testSize': 0.3, 'randomState': 42, 'nIter': int(model.n_iter_[0]),
               'sklearn': __import__('sklearn').__version__, 'nFeatures': int(len(coef))},
    'slots': slots,
    'minutesCoef': beta,
    'intercept': b0,
    'metrics': {'train': metrics(X_train, y_train), 'test': metrics(X_test, y_test)},
    'viz': {
        'roc': {'test': roc(y_test, p_test), 'train': roc(y_train, proba_all[idx_train])},
        'calibration': calib,
        'boundary': {'s': [r4(v) for v in s_grid], 'minutes': [r4(v) for v in m_grid], 'p': grid,
                     'points': [[r4(S[i]), r4(minutes.iloc[i]), int(y.iloc[i])] for i in pts]},
    },
    # Uma partida real do conjunto de teste, para o preset "partida do teste".
    'testMatch': {**row_input(int(idx_test[0])), 'result': data.iloc[int(idx_test[0])]['result']},
}

# ---------- Fixtures: 19 partidas do teste + 1 com campeão desconhecido (handle_unknown='ignore') ----------
fixtures = [{**row_input(int(i)), 'proba': float(proba_all[int(i)])} for i in idx_test[:19]]
odd = {'team1': ['Campeão Novo', 'Graves', 'Yone', 'Jhin', 'Leona'],
       'team2': ['Camille', 'Lillia', 'Akali', "Kai'Sa", 'Campeão Novo'], 'minutes': 31.5}
odd_x = sp.hstack([encoder.transform(pd.DataFrame([odd['team1'] + odd['team2']], columns=SLOTS)), [[odd['minutes']]]])
fixtures.append({**odd, 'proba': float(model.predict_proba(odd_x)[0, 1])})


# ---------- Dashboard (EDA) ----------
def encode(df):
    return sp.hstack([encoder.transform(df[SLOTS]), df[['game_length_minutes']].values]).tocsr()


# Importância por permutação de cada coluna original (as 10 posições + duração), queda de AUC no teste.
test_df = X.iloc[idx_test].reset_index(drop=True)
base_auc = roc_auc_score(y_test, model.predict_proba(encode(test_df))[:, 1])
perm = []
for col in SLOTS + ['game_length_minutes']:
    drops = []
    for k in range(10):
        shuffled = test_df.copy()
        shuffled[col] = np.random.default_rng(k).permutation(shuffled[col].values)
        drops.append(base_auc - roc_auc_score(y_test, model.predict_proba(encode(shuffled))[:, 1]))
    i = SLOTS.index(col) if col in SLOTS else -1
    perm.append({'label': 'Duração' if i < 0 else f'{ROLES[i % 5]} · Time {1 if i < 5 else 2}',
                 'mean': r4(np.mean(drops)), 'std': r4(np.std(drops))})
perm.sort(key=lambda d: -d['mean'])

pop = {s: data[s].map(data[s].value_counts() / len(data)) for s in SLOTS}
numeric = pd.DataFrame({
    'Duração': minutes,
    'Vitória T1': y,
    'Hora do registro': ts.dt.hour,
    'Dia do período': (ts - ts.min()).dt.days,
    'Popularidade T1': sum(pop[s] for s in SLOTS[:5]) / 5,
    'Popularidade T2': sum(pop[s] for s in SLOTS[5:]) / 5,
})
corr = numeric.corr()

edges = np.arange(8, 52, 2)
hist = [{'from': int(a), 'to': int(b),
         'win': int(((minutes >= a) & (minutes < b) & (y == 1)).sum()),
         'loss': int(((minutes >= a) & (minutes < b) & (y == 0)).sum())} for a, b in zip(edges[:-1], edges[1:])]
assert sum(h['win'] + h['loss'] for h in hist) == len(data)

by_len = []
for a, b in [(0, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 60)]:
    m = (minutes > a) & (minutes <= b)
    by_len.append({'from': a, 'to': b, 'rate': r4(y[m].mean()), 'n': int(m.sum())})

week = ts.dt.to_period('W').dt.start_time.dt.strftime('%Y-%m-%d')
timeline = [{'week': w, 'n': int(g.size), 'rate': r4(g.mean())} for w, g in y.groupby(week)]

servers = sorted(({'server': k, 'n': int(g.size), 'rate': r4(g.mean())} for k, g in y.groupby(data['server'])),
                 key=lambda d: -d['n'])

top_picks = []
for i, s in enumerate(SLOTS[:5]):
    vc = data[s].value_counts().head(8)
    top_picks.append({'role': ROLES[i], 'champions': [
        {'champion': c, 'n': int(n), 'rate': r4(y[data[s] == c].mean())} for c, n in vc.items()]})

table = [[int(i), r['server'], r['timestamp'][:10], *[r[s] for s in SLOTS], r4(r['game_length_minutes']), r['result']]
         for i, r in data.head(200).iterrows()]

# ---------- Downloads (servidos pelo site em data/) ----------
model_text = json.dumps(model_json, ensure_ascii=False)
(pub / 'predictwin-lol-s11.csv').write_bytes((root / 'assets' / 's11.csv').read_bytes())
(pub / 'predictwin-modelo.json').write_text(model_text, encoding='utf-8')
with open(pub / 'predictwin-dicionario.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['coluna', 'tipo', 'descricao', 'exemplo'])
    w.writerow(['result', 'texto', 'Resultado da partida: Victory ou Defeat. O código original trata como o resultado do Time 1.', data['result'].iloc[0]])
    w.writerow(['server', 'texto', 'Rótulo de servidor no arquivo (www, euw, na, eune).', data['server'].iloc[0]])
    for i, s in enumerate(SLOTS):
        w.writerow([s, 'texto', f'Campeão do Time {1 if i < 5 else 2} na coluna {i % 5 + 1} (rota inferida: {ROLES[i % 5]}).', data[s].iloc[0]])
    w.writerow(['timestamp', 'data e hora', 'Momento do registro da partida (fuso não documentado).', data['timestamp'].iloc[0]])
    w.writerow(['game_length', 'texto', 'Duração no formato "Xm Ys". O modelo converte para minutos.', data['game_length'].iloc[0]])


def size(n):
    return (pub / n).stat().st_size


downloads = [
    {'file': 'predictwin-lol-s11.csv', 'label': 'Dataset completo (CSV)', 'bytes': size('predictwin-lol-s11.csv'),
     'description': f'{len(data)} partidas reais, 14 colunas. Cópia fiel de assets/s11.csv do repositório original.'},
    {'file': 'predictwin-modelo.json', 'label': 'Modelo exportado (JSON)', 'bytes': size('predictwin-modelo.json'),
     'description': f'Intercepto, {len(coef)} coeficientes por posição e campeão, métricas e curvas. O mesmo arquivo que o simulador usa.'},
    {'file': 'predictwin-dicionario.csv', 'label': 'Dicionário de dados (CSV)', 'bytes': size('predictwin-dicionario.csv'),
     'description': 'Cada coluna do dataset com tipo, descrição e um exemplo.'},
]

eda = {
    'kpis': {'rows': int(len(data)), 'columns': int(data.shape[1] - 1), 'missing': int(data.isna().sum().sum()),
             'duplicates': int(data.duplicated().sum()), 'from': data['timestamp'].min(), 'to': data['timestamp'].max(),
             'champions': int(len(set().union(*[set(data[s]) for s in SLOTS]))), 'victories': int(y.sum()),
             'defeats': int((1 - y).sum()),
             'minutes': {k: r4(v) for k, v in minutes.describe()[['min', '25%', '50%', '75%', 'max', 'mean']].items()}},
    'servers': servers,
    'lengthHist': hist,
    'winByLength': by_len,
    'timeline': timeline,
    'topPicks': top_picks,
    'corr': {'labels': list(corr.columns), 'matrix': [[r4(v) for v in row] for row in corr.values]},
    'permutation': {'metric': 'AUC ROC (teste)', 'base': r4(base_auc), 'repeats': 10, 'items': perm},
    'table': {'columns': ['#', 'Servidor', 'Data', *[f'{ROLES[i % 5]} T{1 if i < 5 else 2}' for i in range(10)], 'Minutos', 'Resultado'],
              'rows': table},
    'downloads': downloads,
}

(out / 'model.json').write_text(model_text, encoding='utf-8')
(out / 'eda.json').write_text(json.dumps(eda, ensure_ascii=False), encoding='utf-8')
(out / 'fixtures.json').write_text(json.dumps(fixtures, ensure_ascii=False, indent=1), encoding='utf-8')
print(f"export: model.json ({len(coef)} coeficientes), eda.json, fixtures.json ({len(fixtures)}), public/data ({len(downloads)} arquivos)")
