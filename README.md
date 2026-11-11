# Vélo Demain 🚲

Prévoir les locations quotidiennes de vélos pour préparer les ressources d’un réseau de mobilité. Projet de portfolio pour candidater à un stage en data science / machine learning.

![Prévisions sur la période de test](reports/forecast.png)

## Voir le projet

- Ouvrir [la démo HTML](reports/index.html) dans un navigateur : aucun serveur nécessaire.
- Consulter [les métriques calculées](reports/metrics.json) et [les prévisions du test](reports/predictions.csv).
- Lancer l’application Streamlit pour filtrer les journées, comparer les modèles et examiner les erreurs.
- Parcourir [le notebook guidé](notebooks/01_exploration.ipynb).

## Installation et exécution

Python 3.12 recommandé. Depuis la racine du dépôt :

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux : source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.pipeline
python -m streamlit run app.py
```

`requirements.txt` fixe les dépendances principales testées. `requirements-lock.txt` fige aussi leurs dépendances transitives dans l’environnement Python 3.12 utilisé lors de la réalisation ; pour le reproduire, installer ce fichier à la place. Le premier entraînement télécharge `day.csv` s’il manque. Les données quotidiennes sont également incluses pour permettre l’exécution hors ligne après installation.

## Question et protocole

À la fin d’une journée, peut-on prévoir le nombre total de locations du lendemain, en utilisant le calendrier et les comptages passés ? La prévision concerne le réseau entier ; elle ne permet pas de répartir des vélos entre stations.

| Partie | Dates | Rôle |
|---|---|---|
| Apprentissage | 8 janvier 2011 – 30 juin 2012 | Ajustement des modèles |
| Validation | 1er juillet – 30 septembre 2012 | Choix du modèle |
| Test | 1er octobre – 31 décembre 2012 | Évaluation finale |

Les sept premiers jours servent à construire les retards. Trois candidats sont comparés sur la validation : la même journée une semaine auparavant, un gradient boosting et une forêt aléatoire. Les paramètres sont fixés dans le code, sans recherche sur le test. Le candidat ayant la plus faible MAE de validation est retenu, y compris si c’est la référence naïve. Un modèle appris retenu est réentraîné sur apprentissage + validation.

Les prédictions de test suivent un protocole **glissant à J+1** : pour chaque journée, les vrais comptages de la veille et de la semaine précédente sont connus. Le modèle reste fixe pendant le trimestre. Cette expérience ne mesure pas une prévision de tout le trimestre effectuée le 30 septembre.

### Fuites de données évitées

- `casual` et `registered` sont exclus : leur somme est exactement `cnt`, la cible.
- Les retards et moyennes mobiles sont décalés avant calcul : aucune cible du jour ne devient une entrée.
- La météo réalisée du lendemain est exclue, car elle n’est pas une prévision disponible la veille.
- L’importance par permutation est calculée sur validation, sur le modèle appris le plus performant de validation.
- Le test ne sert ni au choix des paramètres, ni à la sélection des variables, ni au choix du modèle.

Les tests vérifient notamment qu’une modification des cibles futures ne change aucune variable disponible au moment de prévoir, et que les partitions sont ordonnées et disjointes.

`ci-template.yml` fournit un modèle GitHub Actions pour installer l’environnement, exécuter les tests et lancer le pipeline. La CI automatique n’est pas activée : pour l’activer, déplacer ce fichier vers `.github/workflows/ci.yml` depuis une connexion autorisant l’écriture des workflows.

## Résultats et interprétation

| Évaluation | Modèle | MAE (locations/jour) | RMSE |
|---|---|---:|---:|
| Test final | Gradient boosting, sélectionné sur validation | 1 757 | 2 313 |
| Test final | Même jour de la semaine précédente | 1 458 | 2 047 |

Sur ce test, le modèle sélectionné **dégrade la MAE de 20,5 %** par rapport à la référence. Il ne justifie donc pas de remplacer la référence dans les conditions étudiées. Ce résultat est conservé sans modifier le modèle après consultation du test. L’étape suivante serait une validation sur plusieurs fenêtres temporelles, puis une nouvelle évaluation sur une période encore jamais utilisée.

Le rapport HTML et `reports/metrics.json` contiennent les résultats effectivement calculés, la taille des partitions, les versions des bibliothèques et l’empreinte SHA-256 du CSV. MAE et RMSE sont exprimées en locations par jour. Une amélioration positive de MAE signifie une erreur plus faible que la référence hebdomadaire.

L’application permet d’examiner les erreurs extrêmes. Une importance positive par permutation signifie qu’altérer une variable dégrade la MAE ; cela ne prouve pas une relation causale. Des variables corrélées peuvent se partager ou masquer leur importance.

Les requêtes [analysis.sql](analysis.sql) permettent d’explorer les volumes mensuels et les erreurs à partir de la base SQLite générée dans `reports/analytics.sqlite`.

## Limites et prochaines expériences

Les données datent de 2011–2012 et concernent un seul réseau. Le test couvre un trimestre, avec un contexte différent de la validation ; il ne garantit pas la performance dans d’autres saisons ou sur un réseau actuel. L’erreur moyenne masque les journées atypiques. La tendance temporelle peut mal extrapoler.

Pour aller plus loin : validation à origines glissantes sur plusieurs saisons, intervalles de prédiction calibrés sur une période distincte, comparaison avec des prévisions météo archivées, évaluation à J+7 adaptée aux données disponibles, données récentes et suivi de dérive. Aucun gain financier ni effet sur la disponibilité des vélos n’est établi par cette étude.

## Préparer l’entretien

1. Expliquer le besoin métier et l’unité de la cible.
2. Montrer comment une fuite via `registered` produirait un score trompeur.
3. Justifier la référence saisonnière et le découpage temporel.
4. Comparer les erreurs du modèle et de la référence, y compris les journées où le modèle échoue.
5. Proposer une amélioration mesurable et un protocole pour la tester.

Le code a été préparé avec une assistance IA. Avant de présenter ce projet comme travail personnel, reproduire les résultats, comprendre les choix et ajouter une expérience personnelle documentée. Les mesures du projet peuvent être citées sur un CV une fois vérifiées ; elles ne représentent pas un impact en entreprise.

## Données et attribution

Fanaee-T, H. (2013). *Bike Sharing*. UCI Machine Learning Repository. DOI : [10.24432/C5W894](https://doi.org/10.24432/C5W894). Licence des données : [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Fichier utilisé : `day.csv`, 731 journées du 1er janvier 2011 au 31 décembre 2012. Les retards, variables calendaires dérivées, prévisions et rapports sont des transformations réalisées pour ce projet. Le fichier `Readme.txt` d’origine est conservé dans `data/raw/`. La licence MIT du code ne remplace pas la licence des données.
