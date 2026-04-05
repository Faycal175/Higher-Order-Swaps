# Higher Order Swap Replication

## Volatility Trading Project — Master 272 (Université Paris Dauphine)

Ce projet a été réalisé dans le cadre du cours de **Stratégies de Volatilité** du **Master 272 - Université Paris Dauphine**.

L'objectif est d'étudier, d'implémenter et de backtester la réplication de swaps de moments (Variance, Gamma, Skew, M4) à partir d'options vanilles, puis d'analyser leur interaction avec une stratégie de carry optionnelle.

## Aperçu

Le dépôt combine :

- une base pédagogique issue du cours de volatilité ;
- des développements spécifiques autour des swaps de moments ;
- un notebook principal regroupant théorie, implémentation et résultats empiriques.

## Objectifs du projet

- Construire les noyaux de réplication pour différents contrats de volatilité.
- Backtester des stratégies short swaps, avec et sans coûts de transaction.
- Décomposer le P&L en composantes gamma, theta, vega et résiduelle.
- Étudier la combinaison d'overlays de moments avec une stratégie de carry.
- Analyser les corrélations, drawdowns et comportements en période de stress (COVID) et hors stress.

## Structure du dépôt

```text
investment_lab/        Code source Python du projet
investment_lab/data/   Accès aux données options et taux
data/                  Données locales utilisées par le projet
notebooks/             Notebooks d'analyse et de restitution
README.md              Présentation du projet
requirements.txt       Dépendances Python
pyproject.toml         Métadonnées du package
```

## Installation

### Prérequis

- Python 3.10 ou version supérieure.
- Un environnement virtuel Python est recommandé.

### Installation des dépendances

```bash
pip install -r requirements.txt
pip install -e .
```

## Notebook principal

Le notebook principal du projet est :

`notebooks/Moment_Swaps_Portfolio2.ipynb`

Il contient notamment :

- la partie théorique avec formules et interprétations ;
- la réplication discrète des swaps de moments ;
- les backtests ;
- les analyses de portefeuille (NAV, MDD, corrélations, périodes COVID et hors COVID).

## Modules clés

Les développements spécifiques à ce projet autour des swaps de moments incluent notamment :

- `investment_lab/swap.py`
- `investment_lab/swap_core.py`
- `investment_lab/swap_positions.py`
- `investment_lab/swap_series.py`
- `investment_lab/swap_backtester.py`
- `investment_lab/swap_plotting.py`

Les contributions principales portent sur :

- la refactorisation orientée objet et l'intégration dans le notebook principal ;
- l'ajout des analyses skew et M4 ;
- la prise en compte des coûts bid-ask ;
- la comparaison des overlays ;
- les diagnostics rolling.

## Attribution du code

Ce projet s'appuie sur une base pédagogique issue du cours de **Baptiste Zloch**.

Les modules ci-dessous dans `investment_lab/` proviennent de cette base, ou en sont directement hérités :

- `investment_lab/data/option_db.py`
- `investment_lab/data/rates_db.py`
- `investment_lab/backtest.py`
- `investment_lab/constants.py`
- `investment_lab/dataclass.py`
- `investment_lab/option_selection.py`
- `investment_lab/option_strategies.py`
- `investment_lab/option_trade.py`
- `investment_lab/rates.py`
- `investment_lab/util.py`


## Avertissement

Ce contenu est fourni uniquement à des fins pédagogiques et académiques.

- Il ne constitue pas un conseil en investissement.
- Les résultats de backtest dépendent des hypothèses retenues : données, conventions, coûts, fréquence de rebalancement, etc.
- Les performances passées ne préjugent pas des performances futures.

## Références académiques

- Demeterfi, Derman, Kamal, Zou (1999), *A Guide to Volatility and Variance Swaps*.
- Britten-Jones & Neuberger (2000), *Option Prices, Implied Price Processes, and Stochastic Volatility*.
- Carr & Madan (1998), *Towards a Theory of Volatility Trading*.
- Carr & Lee (2009), *Volatility Derivatives*.

## Citation du cours source

```bibtex
@misc{VolatilityTradingCourse2026,
  title={Volatility Trading Course: Lecture Materials and Code},
  author={Baptiste ZLOCH},
  year={2026},
  howpublished={\url{https://github.com/BaptisteZloch/Volatility-Investment-Course}},
  note={Python implementation of volatility surface modeling, SABR, and SSVI calibration, option strategy backtests, and volatility derivatives.}
}
```

## Licence

Ce dépôt académique est distribué sous licence **CC BY-NC-SA 4.0**, sauf mention contraire.

Pour les éléments hérités de ressources externes, notamment le cours source, se référer à la licence du dépôt d'origine.