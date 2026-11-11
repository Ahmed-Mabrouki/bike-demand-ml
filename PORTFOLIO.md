# Projets de portfolio pour un stage en data science

Profil : niveau intermédiaire en Python et SQL. L’objectif est de pouvoir expliquer un problème, justifier une évaluation et montrer une démo reproductible en entretien. Aucun projet ne garantit l’obtention d’un stage.

| Projet | Question métier | Compétences à montrer | Livrable |
|---|---|---|---|
| Vélo Demain | Combien de locations prévoir le lendemain ? | Séries temporelles, variables retardées, modèles d’arbres, évaluation chronologique, SQL | Projet réalisé dans `bike-demand-ml`, rapport HTML et application Streamlit |
| Campaign Lift | Quels prospects prioriser avant un appel ? | Classification déséquilibrée, régression logistique, boosting, calibration, choix de seuil | Projet proposé : application de simulation d’un budget d’appels |
| Customer Radar | Quels profils de clients et quels risques de non-réachat observe-t-on ? | Préparation transactionnelle, RFM, clustering et stabilité, définition d’une cible future | Projet proposé : exploration de segments et rapport métier |

## 1. Vélo Demain — réalisé

Source : [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

Le dépôt inclut les données quotidiennes, le pipeline, les contrôles temporels, une référence hebdomadaire, deux modèles appris, les métriques réellement calculées, les requêtes SQL et une démo. Les limites du protocole à J+1 sont documentées.

## 2. Campaign Lift — prochaine proposition

Source : [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).

Prévoir la souscription à un dépôt à terme à partir d’un jeu historique de campagnes. Choisir la variante `bank-additional-full.csv`, dont l’ordre chronologique est documenté ; vérifier les variables et leur disponibilité avant l’appel. Exclure notamment la durée de l’appel pour un modèle utilisable avant contact.

Comparer une référence de fréquence, une régression logistique et un modèle d’arbres. Conserver un test temporel final, ajuster le seuil sur validation et mesurer average precision, calibration, précision parmi les premiers prospects et rappel à budget fixé. Présenter un calcul économique comme une simulation fondée sur des hypothèses explicites, sans prétendre mesurer un effet causal d’une campagne.

Ce projet complète le premier avec la classification et une décision métier. La démo permettrait de modifier le nombre d’appels et de comparer les stratégies de sélection. Les données anciennes limitent tout usage bancaire réel.

## 3. Customer Radar — prochaine proposition

Source : [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail).

Nettoyer les transactions, documenter le traitement des annulations et des identifiants clients manquants, puis créer des variables de récence, fréquence et montant à une date de référence. Comparer une segmentation RFM à K-means, en évaluant la stabilité et l’interprétation des segments.

Une extension supervisée définirait précisément la cible : par exemple, absence d’achat dans les 60 jours suivants. Garder une fenêtre d’observation complète pour chaque client et réserver une période future au test. Dans ce contexte, cette cible est un indicateur de non-réachat ; elle ne démontre pas un départ définitif du client.

La démo permettrait de consulter les segments et de proposer une action à tester. Les segments n’établissent pas l’efficacité des actions : une expérimentation serait nécessaire.

## Ordre proposé

Commencer par comprendre et personnaliser Vélo Demain, puis réaliser Campaign Lift pour montrer une deuxième famille de problèmes. Customer Radar sera utile pour des candidatures dans l’e-commerce ou la connaissance client. Mieux vaut deux projets dont tu peux défendre les choix en entretien qu’une accumulation de dépôts que tu n’as pas explorés.
