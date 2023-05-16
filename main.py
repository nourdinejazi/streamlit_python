import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns 
import datetime as dt 

option=st.sidebar.multiselect("**Filtre Dataset**", ["Titanic","Penguins","Trees","Cars","Breast Cancer kaggle dataset"], max_selections=1)



if len(option)>0: 
    if   option[0]=="Titanic" : 
        st.title('Titanic Dataset EDA by Nourdine Jazi TI12 iset Nabeul')
        st.markdown('Ce projet consiste à explorer et analyser les données de ce célèbre accident maritime afin de mieux comprendre les caractéristiques des passagers et les facteurs qui ont influencé leur survie ou leur décès. ')

        data=pd.read_csv("titanic.csv")

        st.write(data.head())


        st.header(" Analyse sur  les variables Numerique : ")

        def plot_num(variable):
            fig = px.histogram(data,x=variable,color_discrete_sequence=['#E97171'] )
            st.plotly_chart(fig)
            st.write('***')
            

        numericVar = ["Age","Fare"]

        for n in numericVar:
            plot_num(n)


        def plot_char(variable):
            varValue = data[variable].value_counts()
            fig = px.histogram(data,x=variable,color_discrete_sequence=['#810000','#E97171'] )
            col1,col2=st.columns((4,1))
            with col1:
                st.plotly_chart(fig)
            with col2 : 
                st.write(varValue)

            st.write('***')


        st.header(" Analyse sur  les variables Categorique : ")
        category1 = ["Sex","Embarked","SibSp", "Parch","Survived","Pclass"]
        for c in category1:
            plot_char(c)



        st.header(" Analyse sur  la variable Sex  : ")


        fig = px.histogram(data,x='Sex',y='Survived',color='Sex',color_discrete_sequence=['#810000','#E97171'] ,title="Nombre  des personnes  Survécu par Sex")
        st.plotly_chart(fig)

        fig = px.histogram(data,x='Sex',color='Pclass',color_discrete_sequence=['#810000','#E97171','#FFCB8E','#D28282','#D3756B','#BB6464'] ,title="Sex et Pclass")
        st.plotly_chart(fig)

        fig = px.histogram(data,x='Sex',color='Embarked',color_discrete_sequence=['#810000','#E97171','#FFCB8E','#D28282','#D3756B','#BB6464'] ,title="Sex et Embarked")
        st.plotly_chart(fig)

        fig = px.histogram(data,x='Age',color='Sex',color_discrete_sequence=['#810000','#E97171','#FFCB8E','#D28282','#D3756B','#BB6464'] ,title="Sex et Embarked")
        st.plotly_chart(fig)
    elif option[0]=="Penguins": 
        st.title("Palmer's Penguins") 
        st.markdown("Statistique sur Palmer's Penguins en utilisant scatterplot! Nourdine Jazi TI12 iset Nabeul") 
        
        selected_x_var = st.selectbox('Varible X ', 
        ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']) 
        selected_y_var = st.selectbox('Variable Y ', 
        ['bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'body_mass_g']) 
        
        penguins_df = pd.read_csv('penguins.csv') 
        sns.set_style('darkgrid')
        markers = {"Adelie": "X", "Gentoo": "s", "Chinstrap":'o'}
        fig, ax = plt.subplots() 
        ax = sns.scatterplot(data = penguins_df, x = selected_x_var, 
        y = selected_y_var, hue = 'species', markers = markers,
        style = 'species') 

        plt.xlabel(selected_x_var) 
        plt.ylabel(selected_y_var) 
        plt.title("Scatterplot de Palmer's Penguins") 
        st.pyplot(fig) 
    elif option[0]=="Trees" :
        st.title('SF Trees Nourdine Jazi TI12 iset Nabeul') 

        st.write("Statistique sur San Francisco trees en utilisant scatterplot! ") 
        trees_df = pd.read_csv('trees.csv') 

        trees_df['age'] = (pd.to_datetime('today') - 

        pd.to_datetime(trees_df['date'])).dt.days 

        owners = st.sidebar.multiselect('**Filtre pour Tree owner**', trees_df['caretaker'].unique()) 

        graph_color = st.sidebar.color_picker('Graph Colors') 

        if owners: 
            trees_df = trees_df[trees_df['caretaker'].isin(owners)]  


        df_dbh_grouped = pd.DataFrame(trees_df.groupby(['dbh']).count()['tree_id']) 

        df_dbh_grouped.columns = ['tree_count'] 

        col1, col2 = st.columns(2) 

        with col1: 

            st.write('Trees by Width') 

            fig_1, ax_1 = plt.subplots() 

            ax_1 = sns.histplot(trees_df['dbh'],  

            color=graph_color) 

            plt.xlabel('Tree Width') 

            st.pyplot(fig_1) 

        with col2: 

            st.write('Trees by Age') 
            fig_2, ax_2 = plt.subplots() 
            ax_2 = sns.histplot(trees_df['age'], 
            color=graph_color) 
            plt.xlabel('Age (Days)') 
            st.pyplot(fig_2) 
        st.write('Trees by Location') 
        trees_df = trees_df.dropna(subset=['longitude', 'latitude']) 
        trees_df = trees_df.sample(n = 1000, replace=True) 
        st.map(trees_df) 

    elif option[0]=="Cars" :
        st.title("Mon notebook sur github statistical analysis and prediction with linear regression using regularization sur Cars dataset.") 
        st.write('https://github.com/nourdinejazi/Machine-Learning-Projects/blob/a5ba0466ba97e4899cdb998436a873a26a594d37/car%20price%20prediction%20-ed1.ipynb')

    elif option[0]=="Breast Cancer kaggle dataset" : 
        st.title('Breast Cancer kaggle dataset statistical analysis and classification with RandomForestClassifier ')
        st.write("https://github.com/nourdinejazi/Machine-Learning-Projects/blob/a5ba0466ba97e4899cdb998436a873a26a594d37/Breast%20Cancer%20kaggle%20dataset.ipynb")
else :
    st.title("Nourdine Jazi TI12")
    st.header(" Institut Supérieur des Etudes Technologiques de Nabeul validation examan TP Python.")
    st.markdown("Plongez dans mes projets variés qui explorent différentes techniques statistiques, de l'analyse exploratoire des données aux modèles prédictifs avancés. ")


