# PredictWin LoL: prevendo vitórias em League of Legends com Regressão Logística 🏆🎮

Trabalho de IA da faculdade que estima a chance de vitória de um time de *League of Legends* a partir dos 5 campeões de cada lado e da duração da partida, agora também como **app de dados ao vivo** (dashboard, avaliação completa e simulador no navegador).

**Ao vivo:** https://g2martins.github.io/PredictWin-LOL-IA/ · **No portfólio:** https://g2portfolio.grovicdata.com/projetos/predictwin-lol-ia

> **Sobre os dados:** são **reais**. O arquivo [`assets/s11.csv`](assets/s11.csv) tem 2.901 partidas ranqueadas registradas entre 25/10/2020 e 31/12/2020 (virada para a Season 11), com servidor, data, os 5 campeões de cada time, duração e resultado. A origem da coleta não foi documentada na época, o arquivo não diz de qual time é o resultado (o código trata como o do Time 1) e só 21,7% das linhas são vitórias, então o rótulo parece vir de um recorte da coleta.

## 📊 Resultado honesto

| Métrica (teste, 871 partidas) | Valor |
|---|---|
| Acurácia | 59,8% |
| Baseline: sempre prever "derrota" | 79,9% |
| Acurácia balanceada | 52,2% |
| AUC ROC | 0,52 (treino: 0,86) |

O modelo decora o treino e quase não generaliza: 777 variáveis (one-hot de campeão por posição + duração) para 2.030 partidas de treino. A versão anterior deste README chamava os 59,7% de "promissor"; comparado ao baseline, não é. A demo mostra isso com curva ROC, calibração, matriz de confusão, importância por permutação e a fronteira de decisão no plano exato do modelo.

## 📂 Arquivos

- **`main.py`**: código original do trabalho (treino, previsão de exemplo e gráfico).
- **`feature.py`**: código original das 10 features mais importantes (continuação do `main.py`).
- **`assets/s11.csv`**: dataset de partidas. **`assets/Sorted_Champions.csv`**: os 153 campeões em ordem alfabética.
- **`export_model.py`**: treina exatamente como o `main.py` e exporta modelo, métricas, curvas, análise exploratória e downloads para o site.
- **`site/`**: o app (Angular 21, pré-renderizado). **`docs/`**: build publicado no GitHub Pages.

## 🚀 Como funciona

1. **Pré-processamento**: a duração (`"30m 44s"`) vira minutos; os 10 campeões passam por `OneHotEncoder(handle_unknown='ignore')` coluna a coluna (Camille no topo é uma variável, Camille no meio é outra).
2. **Treino**: `LogisticRegression(max_iter=5000, C=1, class_weight='balanced')`, split 70/30 com `random_state=42`.
3. **Previsão**: probabilidade de vitória do Time 1 = sigmoide(intercepto + pesos dos campeões + coeficiente × minutos).
4. **No navegador**: `export_model.py` grava os coeficientes em JSON e o site refaz a conta em TypeScript. O build compara 20 previsões com o `predict_proba` do scikit-learn e falha se a diferença passar de 0,000001.

Limitações principais: a duração só existe depois do jogo (não é previsão pré-partida), não há sinergia nem contra-escolha, as probabilidades não são calibradas por causa do `class_weight`, e os dados são de 2020.

### Gráficos do trabalho original

![Probabilidade de Vitória](.github/assets/vitoria.png)
![Importância das Features](.github/assets/feature.png)

## 🐍 Rodar o Python

```bash
pip install -r requirements.txt
cd assets && python ../main.py       # o original lê s11.csv do diretório atual
cd .. && python -I export_model.py   # regenera site/src/app/data/*.json e site/public/data/*
```

## 🌐 Rodar o site

```bash
cd site
npm install
npm start            # http://localhost:4440/PredictWin-LOL-IA/
npm run build:pages  # build + checagem + publica em ../docs/
npm run og           # imagem social e ícones (Playwright em Python)
```

`npm run check` (parte do `build:pages`) valida SEO, seções pré-renderizadas, downloads, ausência de travessão longo e a paridade TypeScript × scikit-learn.

## 🚢 Deploy

GitHub Pages servindo `main:/docs` com base `/PredictWin-LOL-IA/`. O `build:pages` gera `docs/` com `.nojekyll` e `404.html`.

## 🏅 Créditos

Este projeto foi desenvolvido com a colaboração de uma equipe de grandes invocadores! 🌟

- **Caio Silveira Guimarães** 🧠✨
- **Gustavo Martins Gripaldi** 🎮💡
- **Giovanna Monteiro Bispo** 🔍🎯
- **Mariana Monteiro Bispo** 💻🔥
- **Mayron Malaquias Oliveira** ⚡🎨

App de dados por [Gustavo Martins Gripaldi](https://g2portfolio.grovicdata.com). PredictWin não é endossado pela Riot Games e não reflete a visão ou opinião da Riot Games. League of Legends e Riot Games são marcas da Riot Games, Inc.
