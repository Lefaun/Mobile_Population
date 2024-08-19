import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Portugal Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# Load data
df_reshaped = pd.read_csv('data/us-population-2010-2019-reshaped.csv')

# Sidebar
with st.sidebar:
    st.title('🏂 Portugal Population Dashboard')
   
    year_list = list(df_reshaped.year.unique())[::-1]
    selected_year = st.selectbox('Selecione o ano', year_list)
   
    df_selected_year = df_reshaped[df_reshaped.year == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="population", ascending=False)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Selecione a cor do tema', color_theme_list)

# Define functions

def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Ano", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                            legend=None,
                            scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )
    return heatmap

def make_donut(input_response, input_text, input_color):
    color_map = {
        'blue': ['#29b5e8', '#155F7A'],
        'green': ['#27AE60', '#12783D'],
        'orange': ['#F39C12', '#875A12'],
        'red': ['#E74C3C', '#781F16']
    }
    chart_color = color_map.get(input_color, ['#29b5e8', '#155F7A'])
   
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })
   
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Topic:N", scale=alt.Scale(domain=[input_text, ''], range=chart_color), legend=None),
    ).properties(width=130, height=130)
   
    text = plot.mark_text(align='center', color=chart_color[0], font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    return plot + text

def calculate_population_difference(input_df, input_year):
    selected_year_data = input_df[input_df['year'] == input_year].reset_index(drop=True)
    previous_year_data = input_df[input_df['year'] == input_year - 1].reset_index(drop=True)
    selected_year_data['population_difference'] = selected_year_data.population.sub(previous_year_data.population, fill_value=0)
    return pd.concat([selected_year_data.states, selected_year_data.id, selected_year_data.population, selected_year_data.population_difference], axis=1).sort_values(by="population_difference", ascending=False)

def simulate_population_step(population, birth_rate, death_rate):
    births = np.random.poisson(birth_rate)
    deaths = np.random.poisson(death_rate)
    new_population = population + (births - deaths)
    return new_population, births, deaths

def compute_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    variance = np.var(data)
    return mean, std_dev, variance

def perform_regression(time, population):
    X = sm.add_constant(time)
    model = sm.OLS(population, X).fit()
    return model

# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.markdown('#### Gains/Losses')

    df_population_difference_sorted = calculate_population_difference(df_reshaped, selected_year)

    if selected_year > 2010:
        st.metric(label=df_population_difference_sorted.states.iloc[0],
                  value=format_number(df_population_difference_sorted.population.iloc[0]),
                  delta=format_number(df_population_difference_sorted.population_difference.iloc[0]))

        st.metric(label=df_population_difference_sorted.states.iloc[-1],
                  value=format_number(df_population_difference_sorted.population.iloc[-1]),
                  delta=format_number(df_population_difference_sorted.population_difference.iloc[-1]))
    else:
        st.metric(label='-', value='-', delta='')

    st.markdown('#### States Migration')

    if selected_year >= 2010:
        df_greater_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference > 50000]
        df_less_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference < -50000]
       
        states_migration_greater = round((len(df_greater_50000)/df_population_difference_sorted.states.nunique())*100)
        states_migration_less = round((len(df_less_50000)/df_population_difference_sorted.states.nunique())*100)

        st.write('Births')
        st.altair_chart(make_donut(states_migration_greater, 'Births', 'green'))
        st.write('Deaths')
        st.altair_chart(make_donut(states_migration_less, 'Deaths', 'red'))
    else:
        st.write("No data available for selected year.")

with col[1]:
    st.title("Simulação de População com Atualizações ao Vivo")

    initial_population = st.number_input("População Inicial", value=1000, min_value=1)
    birth_rate = st.slider("Taxa de Nascimento (por segundo)", 0.0, 5.0, 1.0)
    death_rate = st.slider("Taxa de Mortalidade (por segundo)", 0.0, 5.0, 0.5)
    seconds = st.number_input("Duração da Simulação (segundos)", value=100, min_value=1)
   
    if st.button("Iniciar Simulação"):
        time_data = []
        population_data = []
        births_data = []
        deaths_data = []

        population = initial_population
        placeholder = st.empty()
   
        for second in range(seconds):
            population, births, deaths = simulate_population_step(population, birth_rate, death_rate)
           
            time_data.append(second)
            population_data.append(population)
            births_data.append(births)
            deaths_data.append(deaths)

            with placeholder.container():
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric(label="População Atual", value=int(population))
                kpi2.metric(label="Nascimentos no último segundo", value=int(births))
                kpi3.metric(label="Mortes no último segundo", value=int(deaths))
     
                st.markdown("### Evolução da População")
                st.line_chart(pd.DataFrame({"Tempo": time_data, "População": population_data}).set_index('Tempo'))

                st.markdown("### Nascimentos e Mortes")
                st.area_chart(pd.DataFrame({"Tempo": time_data, "Nascimentos": births_data, "Mortes": deaths_data}).set_index('Tempo'))

            time.sleep(1)  # Esperar um segundo antes de atualizar novamente

        model = perform_regression(time_data, population_data)
        st.write(model.summary())
       
        st.markdown("### Regressão Linear da População")
        df = pd.DataFrame({"Tempo": time_data, "População": population_data})
        df['Previsão'] = model.predict(sm.add_constant(time_data))
        st.line_chart(df[['Tempo', 'População', 'Previsão']].set_index('Tempo'))

    # Heatmap


Paulo Monteiro <vegaspace@gmail.com>
10:33 AM (0 minutes ago)
to me

# Heatmap function
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                            legend=None,
                            scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        )
    return heatmap

# Main dashboard section for the heatmap
with col[1]:
    # Assuming the selected year is already filtered
    st.markdown("### Heatmap of Population by State and Year")
    heatmap = make_heatmap(df_reshaped, 'year', 'states', 'population', selected_color_theme)
    st.altair_chart(heatmap, use_container_width=True)
