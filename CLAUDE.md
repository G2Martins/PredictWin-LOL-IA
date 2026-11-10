# CLAUDE.md, PredictWin LoL

> **Spec viva.** Porta de entrada lida pelo agente em toda sessão. Não duplica o [README.md](README.md), **aponta** pra ele. Atualize quando uma regra, comando, módulo ou hurdle mudar.
>
> **Última revisão:** 2026-11-10
> **Dados, métricas, como rodar e deploy:** [README.md](README.md)

---

## 1. O que é

Trabalho de IA da faculdade (regressão logística sobre 2.901 partidas **reais** de League of Legends, fim de 2020) transformado em **app de dados** de portfólio de **Gustavo Martins Gripaldi**: visão geral, dashboard exploratório, avaliação do modelo, simulador e downloads. Linkado em https://g2portfolio.grovicdata.com/projetos/predictwin-lol-ia. Ao vivo: https://g2martins.github.io/PredictWin-LOL-IA/. O tom é **honesto**: o modelo perde para o baseline e a página diz isso.

## 2. Stack + Comandos

Python (pandas, scikit-learn) na raiz; site em `site/`: Angular 21.2 standalone + zoneless + signals, SSG (`outputMode: "static"`, uma rota, sem router), `provideClientHydration(withEventReplay())`, Tailwind v4, Geist + Geist Mono, Node 22 (scripts `.ts` nativos). GitHub Pages em `main:/docs`, `baseHref` `/PredictWin-LOL-IA/`.

```bash
python -I export_model.py        # (raiz) treina como main.py e gera site/src/app/data/*.json + site/public/data/*
cd site
npm start                        # http://localhost:4440/PredictWin-LOL-IA/
npm run build                    # ng build com prerender
npm run check                    # smoke pós-build: SEO, seções, downloads, sem travessão longo, paridade TS x sklearn
npm run build:pages              # build + check + copia para ../docs (.nojekyll, 404.html)
npm run og                       # python scripts/og.py: OG 1200x630 + ícones PNG via Playwright
```

## 3. Mapa de módulos

| Caminho | Responsabilidade |
|---|---|
| `main.py`, `feature.py` | Código original do trabalho. Não mexer. |
| `assets/s11.csv` | Dataset real. `assets/Sorted_Champions.csv`: lista de campeões. |
| `export_model.py` | Única ponte Python → site: treino idêntico ao `main.py`, métricas, ROC, calibração, plano de decisão, EDA, permutação, tabela de 200 linhas, fixtures e downloads. |
| `site/src/app/data/` | `model.json` (coeficientes com precisão total + métricas + curvas), `eda.json` (dashboard + manifesto de downloads), `fixtures.json` (20 `predict_proba` do sklearn). Gerados, não editar. |
| `site/src/app/model.ts` | Inferência (sigmoide sobre one-hot por posição). Sem imports: `scripts/check.ts` roda o mesmo arquivo no Node. |
| `site/src/app/app.ts` + `app.html` | A página inteira: seções Visão geral, Dashboard, Modelo, Simulador, Dados. Gráficos em SVG/CSS, sem biblioteca. |
| `site/public/data/` | Downloads servidos pelo site (CSV completo, modelo, dicionário). Gerados pelo export. |
| `site/scripts/` | `check.ts`, `pages.ts`, `og.py` + `og.html`. |
| `docs/` | Saída do build publicada. Nunca editar à mão. |

## 4. Regras SEMPRE-ATIVAS

- **Honestidade dos dados:** dados reais, origem não documentada; sempre comparar com o baseline (79,9%) e mostrar AUC de teste. Nada de chamar o modelo de "bom".
- **Paridade:** toda mudança no modelo passa por `export_model.py`; o `check` falha se TS e sklearn diferirem mais que 1e-6.
- **Tudo pré-calculado no Python;** o navegador só desenha e faz a soma do simulador.
- **SSG:** APIs do browser só em `afterNextRender` ou eventos. Conteúdo visível sem JS (gate `.js`).
- **Animação só `transform`/`opacity`**, com `prefers-reduced-motion`.
- **Design Valerium:** Geist, eyebrow mono, card "well", ticks, dot grid. Cores só por tokens em `styles.css` (escuro padrão + claro via `prefers-color-scheme`). Azul = Time 1, vermelho = Time 2 (validados com o script do skill `dataviz`), dourado = acento.
- **Zero travessão longo (U+2014)** em texto visível e downloads; o `check` varre html/js/json/csv.
- **Disclaimer** "Projeto de portfólio de Gustavo Martins Gripaldi." no topo e crédito no footer; nota de marca da Riot.
- **`npm run build:pages` verde** antes de publicar. **`docs/` é build.**

## 5. Segurança

- Site estático, sem backend, sem formulário, sem cookies, sem `localStorage`.
- Texto só via interpolação do Angular (escapado); nada de `innerHTML`. JSON-LD é estático no `index.html`.
- Nenhum segredo no repo; `.gitignore` cobre `.env*`, `node_modules`, `dist`, `.angular`, `.shots/`, `venv`.

## 6. Common Hurdles

- **Python com argv de caminho não ASCII ("Programação") quebra:** rodar sempre de dentro da pasta com caminho relativo e `python -I` (`python -I export_model.py`, `python -I scripts/og.py`).
- **Node 22 `cpSync` sai com 127 em silêncio** no caminho com "Programação": `pages.ts` usa `cp` de `fs/promises`.
- **EBUSY** ao recriar `docs/` se um shell ou servidor estiver com cwd dentro dela.
- **Git Bash converte `/PredictWin-LOL-IA/` em caminho do Windows** ao passar para um servidor local: usar `MSYS_NO_PATHCONV=1`.
- **JSON grande tipado pelo TS** (`slots` com 153 campeões) estoura índice: `app.ts` faz cast para `LolModel`.
- **`<option [selected]>` não vai para o HTML pré-renderizado:** usar também `[attr.selected]`.
- **Kaspersky intercepta o `chrome.exe`** (injeta script http e tira o gzip do HTML), derrubando performance e boas práticas no Lighthouse local: rodar o Lighthouse pelo `chrome-headless-shell.exe` do Playwright (`--remote-debugging-port` + `lighthouse --port`), que não é interceptado, com servidor gzip. Assim deu 98/100/100/100 (mobile).
