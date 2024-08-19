#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import urllib.request
#######################
# Page configuration
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import numpy as np
import plotly.express as px # interactive charts 

st.set_page_config(
    page_title="Portugal Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


#######################
# Load data
df_reshaped = pd.read_csv('data/us-population-2010-2019-reshaped.csv')


#######################
# Sidebar
with st.sidebar:
    st.title('🏂 Portugal Population Dashboard')
    
    year_list = list(df_reshaped.year.unique())[::-1]
    
    selected_year = st.selectbox('Selecione o ano', year_list)
    df_selected_year = df_reshaped[df_reshaped.year == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="population", ascending=False)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Selecione a cor do tema', color_theme_list)


#######################
# Plots

# Heatmap
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Ano", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
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
    # height=300
    return heatmap

# Choropleth map
#def make_choropleth(input_df, input_id, input_column, input_color_theme):
    #choropleth = px.choropleth(input_df, locations=input_id, color=input_column, locationmode="USA-states",
                               #color_continuous_scale=input_color_theme,
                               #range_color=(0, max(df_selected_year.population)),
                               #scope="usa",
                               #labels={'population':'Population'}
                              #)
    #choropleth.update_layout(
        #template='plotly_dark',
        #plot_bgcolor='rgba(0, 0, 0, 0)',
        #paper_bgcolor='rgba(0, 0, 0, 0)',
        #margin=dict(l=0, r=0, t=0, b=0),
        #height=350
    #)
    #return choropleth


# Donut chart
def make_donut(input_response, input_text, input_color):
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']
    
  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
  return plot_bg + plot + text

# Convert population to text 
def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'

# Calculation year-over-year population migrations
def calculate_population_difference(input_df, input_year):
  selected_year_data = input_df[input_df['year'] == input_year].reset_index()
  previous_year_data = input_df[input_df['year'] == input_year - 1].reset_index()
  selected_year_data['population_difference'] = selected_year_data.population.sub(previous_year_data.population, fill_value=0)
  return pd.concat([selected_year_data.states, selected_year_data.id, selected_year_data.population, selected_year_data.population_difference], axis=1).sort_values(by="population_difference", ascending=False)

def simulate_population_step(population, birth_rate, death_rate):
    births = np.random.poisson(birth_rate)
    deaths = np.random.poisson(death_rate)
    net_change = births - deaths
    new_population = population + net_change
    
    return new_population, births, deaths

# Função para calcular as estatísticas
def compute_statistics(data):
    mean = np.mean(data)
    #mode = stats.mode(data)[0][0]
    std_dev = np.std(data)
    variance = np.var(data)
    
    return mean, std_dev, variance

# Função para realizar a regressão linear
def perform_regression(time, population):
    X = sm.add_constant(time)
    model = sm.OLS(population, X).fit()
    return model


#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.markdown('#### Gains/Losses')

    df_population_difference_sorted = calculate_population_difference(df_reshaped, selected_year)

    if selected_year > 2010:
        first_state_name = df_population_difference_sorted.states.iloc[0]
        first_state_population = format_number(df_population_difference_sorted.population.iloc[0])
        first_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[0])
    else:
        first_state_name = '-'
        first_state_population = '-'
        first_state_delta = ''
    st.metric(label=first_state_name, value=first_state_population, delta=first_state_delta)

    if selected_year > 2010:
        last_state_name = df_population_difference_sorted.states.iloc[-1]
        last_state_population = format_number(df_population_difference_sorted.population.iloc[-1])   
        last_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[-1])   
    else:
        last_state_name = '-'
        last_state_population = '-'
        last_state_delta = ''
    st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)

    
    st.markdown('#### States Migration')

    if selected_year >= 2010:
        # Filter states with population difference > 50000
        # df_greater_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference_absolute > 50000]
        df_greater_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference > 50000]
        df_less_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference < -50000]
        
        

        
        # % of States with population difference > 50000
        #states_migration_greater= kpi2.metric(label="Nascimentos no último segundo", value=int(births))
        #states_migration_less = kpi3.metric(label= "Mortes no último segundo", value=int(deaths))
        states_migration_greater = round((len(df_greater_50000)/df_population_difference_sorted.states.nunique())*100)
        states_migration_less = round((len(df_less_50000)/df_population_difference_sorted.states.nunique())*100)
        #donut_chart_greater = make_donut(births_data, 'Births', 'green')
        donut_chart_greater = make_donut(states_migration_greater, 'Births', 'green')
        #donut_chart_less = make_donut(deaths_data, 'Deaths', 'red')
        donut_chart_less = make_donut(states_migration_less, 'Deaths', 'red')
    else:
        states_migration_greater = 0
        states_migration_less = 0
        #donut_chart_greater = make_donut(births_data, 'Births', 'green')
        donut_chart_greater = make_donut(states_migration_greater, 'Births', 'green')
        #donut_chart_less = make_donut(deaths_data, 'Deaths', 'red')
        #donut_chart_greater = make_donut(states_migration_greater, 'Inbound Migration', 'green')
        donut_chart_less = make_donut(states_migration_less, 'Deaths', 'red')

    migrations_col = st.columns((0.2, 1, 0.2))
    with migrations_col[1]:
        st.write('Births')
        st.altair_chart(donut_chart_greater)
        st.write('Deaths')
        st.altair_chart(donut_chart_less)

with col[1]:

    data = pd.DataFrame({
    'awesome cities' : ['Portugal','Algarve','Guimarães','Coimbra'],
    'lat' : [38.44,37.01,41.44,40.21],
    'lon' : [ -9.8,-7.9,-8.29,-8.22]
})
    st.map(data)

#######################################################################################
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
        births_Total = []
        deaths_Total =[]
        
        population = initial_population
        
        placeholder = st.empty()
    
        for second in range(seconds):
            population, births, deaths = simulate_population_step(population, birth_rate, death_rate)
            
            time_data.append(second)
            population_data.append(population)
            births_data.append(births)
            deaths_data.append(deaths)
            births_Total.append(birth_rate)
            deaths_Total.append(death_rate)
            Total_Nascimentos = sum(births_Total)
            Total_Mortes = sum(deaths_Total)
            #Totalidade += birth_rate
            mean, std_dev, variance = compute_statistics(population_data)
            
            with placeholder.container():
                # KPIs
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric(label="População Atual", value=int(population))
                kpi2.metric(label="Nascimentos no último segundo", value=int(births))
                kpi3.metric(label="Mortes no último segundo", value=int(deaths))
     
                kpi4, kpi5 = st.columns(2)
                kpi4.metric(label="Total Nascimentos", value=int(Total_Nascimentos))
                kpi5.metric(label="Total Mortes", value=int(Total_Mortes))
    
                # Dados em DataFrame
                df = pd.DataFrame({
                    "Tempo": time_data,
                    "População": population_data,
                    "Nascimentos": births_data,
                    "Mortes": deaths_data
                })
    
                # Gráficos interativos
                st.markdown("### Evolução da População")
                st.line_chart(df[['Tempo', 'População']].set_index('Tempo'))
    
                st.markdown("### Nascimentos e Mortes")
                st.area_chart(df[['Tempo', 'Nascimentos', 'Mortes']].set_index('Tempo'))
    
                st.markdown("### Dados Detalhados")
                st.dataframe(df)
    
                # Estatísticas
                st.write(f"Média da População: {mean}")
                st.write(f"Desvio Padrão da População: {std_dev}")
                st.write(f"Variância da População: {variance}")
                #st.write(f"Total de Nascimento: {Totalidade}")
                st.write(f"Total de Nascimento: {Total_Nascimentos}")
                st.write(f"Total de Mortes: {Total_Mortes}")
            time.sleep(1)  # Esperar um segundo antes de atualizar novamente
    
        model = perform_regression(time_data, population_data)
        st.write(model.summary())
        
        st.markdown("### Regressão Linear da População")
        df['Previsão'] = model.predict(sm.add_constant(time_data))
        st.line_chart(df[['Tempo', 'População', 'Previsão']].set_index('Tempo'))
    ###################################################################################
    #st.title("Simulação de População com Atualizações ao Vivo")

    #initial_population = st.number_input("População Inicial", value=1000, min_value=1)
    #birth_rate = st.slider("Taxa de Nascimento (por segundo)", 0.0, 5.0, 1.0)
    #death_rate = st.slider("Taxa de Mortalidade (por segundo)", 0.0, 5.0, 0.5)
    #seconds = st.number_input("Duração da Simulação (segundos)", value=100, min_value=1)
    
    
        population = initial_population
        
        #for second in range(seconds):
            #population, births, deaths = simulate_population_step(population, birth_rate, death_rate)
            
            #time_data.append(second)
            #population_data.append(population)
            #births_data.append(births)
            #deaths_data.append(deaths)
        mean,  std_dev, variance = compute_statistics(population_data)
            
            #st.write(f"Tempo: {second + 1}s")
            #st.write(f"População Atual: {population}")
            #st.write(f"Nascimentos no último segundo: {births}")
            #st.write(f"Mortes no último segundo: {deaths}")
            #st.write(f"Média da População: {mean}")
           # st.write(f"Moda da População: {mode}")
            #st.write(f"Desvio Padrão da População: {std_dev}")
            #st.write(f"Variância da População: {variance}")
    
        #Atualizar gráficos
        df = pd.DataFrame({
            "Tempo": time_data,
            "População": population_data,
            "Nascimentos": births_data,
            "Mortes": deaths_data
          })

        fig, ax = plt.subplots()
        sns.lineplot(x='Tempo', y='População', data=df, ax=ax, label='População')
        sns.lineplot(x='Tempo', y='Nascimentos', data=df, ax=ax, label='Nascimentos')
        sns.lineplot(x='Tempo', y='Mortes', data=df, ax=ax, label='Mortes')

        ax.set_title('Simulação de População ao Vivo')
        ax.legend()

        st.pyplot(fig)
        
        time.sleep(1)  # Esperar um segundo antes de atualizar novamente

        model = perform_regression(time_data, population_data)
        st.write(model.summary())
    
        fig, ax = plt.subplots()
        sns.regplot(x='Tempo', y='População', data=df, ax=ax, label='População', line_kws={"color":"r","alpha":0.7,"lw":2})

        ax.set_title('Regressão Linear da População')
        ax.legend()

        st.pyplot(fig)

            #######################################################################################
    
    #choropleth = make_choropleth(df_selected_year, 'states_code', 'population', selected_color_theme)
    #st.plotly_chart(choropleth, use_container_width=True)
    ################################################################################################3
    heatmap = make_heatmap(df_reshaped, 'year', 'states', 'population', selected_color_theme)
    st.altair_chart(heatmap, use_container_width=True)



# Função para simular um segundo da população

# Configuração da interface do Streamlit

    

with col[2]:
    st.markdown('#### Top States')

    st.dataframe(df_selected_year_sorted,
                 column_order=("states", "population"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "states": st.column_config.TextColumn(
                        "States",
                    ),
                    "population": st.column_config.ProgressColumn(
                        "Population",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_year_sorted.population),
                     )}
                 )
    
    with st.expander('About', expanded=True):
        st.write('''
            - Data: [Portugal. Census INE - Instituto Nacional de Estatística e PORDATA ](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html).
            - :orange[**Nascimentos/Mortes**]: Demografia Nascimentos e Mortes por Ano atualizados ao segundo
            - :orange[**População por Localidade**]: Percentagem da distribuição Geográfica por Localidade
            ''')

# Função para simular um segundo da população
def simulate_population_step(population, birth_rate, death_rate):
    births = np.random.poisson(birth_rate)
    deaths = np.random.poisson(death_rate)
    net_change = births - deaths
    new_population = population + net_change
    
    return new_population, births, deaths

# Função para calcular as estatísticas
def compute_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    variance = np.var(data)
    
    return mean, std_dev, variance

# Função para realizar a regressão linear
def perform_regression(time, population):
    X = sm.add_constant(time)
    model = sm.OLS(population, X).fit()
    return model

# Configuração da interface do Streamlit
#st.set_page_config(page_title="Simulação de População em Tempo Real", layout="wide")

