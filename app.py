"""Explore the frozen, out-of-time evaluation; no invented model results."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Vélo Demain', page_icon='🚲', layout='wide')
st.title('Vélo Demain')
st.write('Prévision quotidienne de la demande · portfolio data science')
if not (ROOT / 'reports/metrics.json').exists():
    st.info('Lance python -m src.pipeline pour produire les résultats.')
    st.stop()
metrics = json.loads((ROOT / 'reports/metrics.json').read_text(encoding='utf-8'))
df = pd.read_csv(ROOT / 'reports/predictions.csv', parse_dates=['dteday'])
a, b, c = st.columns(3)
a.metric('MAE test', f'{metrics["test"]["mae"]:.0f} locations/jour')
b.metric('Gain de MAE vs semaine précédente', f'{metrics["mae_improvement_percent"]:+.1f} %')
c.metric('Modèle sélectionné sur validation', metrics['selected'])
st.caption('Test : octobre–décembre 2012. Un gain positif indique une amélioration.')
selected = st.select_slider('Période affichée', options=df.dteday.dt.date.tolist(),
                          value=(df.dteday.min().date(), df.dteday.max().date()))
view = df[df.dteday.dt.date.between(*selected)]
st.line_chart(view.set_index('dteday')[['cnt', 'prediction', 'lag_7']].rename(
    columns={'cnt': 'Observé', 'prediction': 'Modèle', 'lag_7': 'Semaine précédente'}))
st.subheader('Comparer les modèles sur validation')
st.dataframe(pd.DataFrame(metrics['validation']).T.rename(columns={'mae': 'MAE', 'rmse': 'RMSE'}))
st.subheader('Quelles variables aident le modèle ?')
st.caption('Permutation sur validation du meilleur modèle appris. Une valeur positive signifie que mélanger cette variable augmente la MAE. Les variables corrélées compliquent l’interprétation.')
st.bar_chart(pd.DataFrame(metrics['validation_importance']).set_index('feature')['mae_increase'])
st.subheader('Les journées les plus difficiles')
st.dataframe(view.nlargest(10, 'absolute_error'), hide_index=True)
st.download_button('Télécharger les prévisions du test', df.to_csv(index=False),
                   file_name='bike_predictions.csv', mime='text/csv')
st.info('Prévision glissante à J+1 : les totaux réels de la veille sont disponibles. '
        'Aucune météo observée du jour prédit. Ce protocole ne mesure pas une prévision à trois mois.')
with st.expander('Limites et provenance'):
    st.write('Données anciennes (2011–2012), un seul réseau, aucun détail par station. '
             'La validation ne couvre qu’un trimestre. Les importance de variables ne prouvent pas de causalité. '
             'Avant un usage réel : backtests sur plusieurs saisons, estimation d’incertitude, '
             'surveillance de dérive et données récentes.')
    st.markdown('[Fanaee-T (2013), UCI Bike Sharing](https://doi.org/10.24432/C5W894), CC BY 4.0.')
