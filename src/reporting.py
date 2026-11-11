"""Local report requiring no web assets or external chart services."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(report, predictions, raw, folder):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 11})
    fig, ax = plt.subplots(figsize=(12, 4.5), layout='constrained')
    ax.plot(predictions.dteday, predictions.cnt, color='#162f39', label='Locations observées', linewidth=2)
    ax.plot(predictions.dteday, predictions.prediction, color='#d05b31', label='Modèle sélectionné', linewidth=1.7)
    ax.set(ylabel='Locations / jour', title='Prévision à J+1 · octobre–décembre 2012')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=.12)
    fig.savefig(folder / 'forecast.png', dpi=160)
    plt.close(fig)
    gain = report['mae_improvement_percent']
    rows = ''.join(f'<tr><td>{name}</td><td>{m["mae"]:.0f}</td><td>{m["rmse"]:.0f}</td></tr>'
                   for name, m in report['validation'].items())
    points = predictions.assign(dteday=predictions.dteday.dt.strftime('%Y-%m-%d')).to_dict(orient='records')
    html = '''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vélo Demain — portfolio ML</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f3ed;color:#19343d;font:16px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:50px 24px}header{border-top:5px solid #d05b31;padding-top:20px}.eyebrow{font:600 12px monospace;letter-spacing:2px;color:#98502e}h1{font-size:clamp(42px,7vw,72px);line-height:1.05;margin:25px 0}h2{font-size:24px}p{max-width:780px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:32px 0}.card{background:#fff;border:1px solid #deded5;padding:22px;border-radius:10px}.number{display:block;font-size:36px;font-weight:650}.muted{color:#596d71;font-size:14px}img{width:100%;border-radius:8px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}section{margin:38px 0}.note{padding:18px 22px;background:#e8ece7;border-left:3px solid #477968}a{color:#a94424}input{width:100%;accent-color:#d05b31}footer{margin-top:40px;border-top:1px solid #ccc;padding-top:20px}@media(max-width:650px){.grid{grid-template-columns:1fr}main{padding:25px 18px}.number{font-size:30px}}
</style><main><header><span class="eyebrow">PORTFOLIO DATA SCIENCE / 01</span><h1>Combien de vélos<br>demain ?</h1><p>Prévoir les locations quotidiennes pour aider une équipe d’exploitation à préparer ses ressources. Une étude historique à l’échelle du réseau Capital Bikeshare.</p></header>
<div class="grid"><div class="card"><span class="muted">Erreur absolue moyenne · test</span><span class="number">__MAE__</span>locations par jour</div><div class="card"><span class="muted">Évolution de MAE vs référence</span><span class="number">__GAIN__ %</span>un gain positif indique une amélioration</div><div class="card"><span class="muted">Période de test séparée</span><span class="number">__DAYS__ jours</span>octobre → décembre 2012</div></div>
<section class="card"><h2>Prévu / observé</h2><img src="forecast.png" alt="Locations observées et prédites sur le dernier trimestre 2012"><p class="muted">Choisis un jour du test pour examiner une prévision historique.</p><input id="day" aria-label="Jour du test" type="range" min="0" max="__MAX__" value="0"><p id="detail" aria-live="polite"></p></section>
<section><h2>Choisir le modèle avant le test</h2><p>Apprentissage jusqu’à juin 2012, validation de juillet à septembre. Le modèle retenu est <strong>__MODEL__</strong>. Il est ensuite réentraîné sur l’apprentissage et la validation ; le dernier trimestre sert à l’évaluation finale.</p><table><thead><tr><th>Validation</th><th>MAE ↓</th><th>RMSE ↓</th></tr></thead><tbody>__ROWS__</tbody></table></section>
<section class="note"><strong>Ce que cette démo permet de conclure</strong><p>Chaque jour est prédit avec les locations réellement connues jusqu’à la veille. Il ne s’agit pas de prévoir trois mois d’un seul coup. Les variables météo observées sont exclues, tout comme les colonnes casual et registered dont la somme est la cible.</p><p>Deux ans de données historiques ne démontrent pas une performance actuelle. Les événements exceptionnels, les évolutions du réseau et l’absence de données par station limitent l’usage opérationnel. Les performances ne prouvent aucune économie financière.</p></section>
<section><h2>À discuter en entretien</h2><ul><li>Pourquoi un découpage aléatoire surestimerait-il la généralisation temporelle ?</li><li>Pourquoi comparer le modèle à la même journée de la semaine précédente ?</li><li>Comment adapter l’évaluation à un horizon de sept jours sans connaître les valeurs intermédiaires ?</li><li>Pourquoi les erreurs les plus importantes méritent-elles une analyse métier ?</li></ul></section>
<footer class="muted">Données : Fanaee-T, H. (2013), <a href="https://doi.org/10.24432/C5W894">Bike Sharing · UCI</a>, CC BY 4.0. Données quotidiennes utilisées ; variables temporelles et résultats dérivés. Rapport généré par <code>python -m src.pipeline</code>.</footer></main>
<script>const days=__DATA__;const day=document.querySelector('#day');function show(){const d=days[Number(day.value)];document.querySelector('#detail').textContent=`${d.dteday} · observé : ${d.cnt} · prévu : ${Math.round(d.prediction)} · erreur : ${Math.round(d.absolute_error)} locations`;}day.addEventListener('input',show);show();</script></html>'''
    for key, value in {'MAE': f'{report["test"]["mae"]:.0f}', 'GAIN': f'{gain:+.1f}',
                        'DAYS': str(len(predictions)), 'MAX': str(len(predictions)-1),
                        'MODEL': report['selected'], 'ROWS': rows,
                        'DATA': json.dumps(points, ensure_ascii=False)}.items():
        html = html.replace('__' + key + '__', value)
    (folder / 'index.html').write_text(html, encoding='utf-8')

