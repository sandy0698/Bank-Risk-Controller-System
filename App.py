import pandas as pd 
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import re
import box
from sklearn.preprocessing import LabelEncoder
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import RetrievalQA




#df_modified = pd.read_csv("E:\\Final Project\\loan_data.csv")
df = pd.read_csv(r"E:\\Final Project\\loan_data.csv")

# --------------------------------------------------Logo & details on top

st.set_page_config(page_title= "Bank Risk Controller System | By Santhosh S",
                   layout= "wide",
                   initial_sidebar_state= "expanded")
image_link = "https://www.relofirm.com/wp-content/uploads/2013/11/panama-bank-account.jpg"
st.markdown(f""" <style>.stApp {{
                    background: url('{image_link}');   
                    background-size: cover}}
                 </style>""",unsafe_allow_html=True)

# Function to safely convert to sqrt
def log_trans(value):
    try:
        return np.log(float(value))  # Conversion to float
    except (ValueError, TypeError):
        raise ValueError(f"Invalid input: {value}")

# Define occupation types in alphabetical order with corresponding numeric codeslabel_encoding
occupation = {
    0: 'Accountants',
    1: 'Cleaning staff',
    2: 'Cooking staff',
    3: 'Core staff',
    4: 'Drivers',
    5: 'HR staff',
    6: 'High skill tech staff',
    7: 'IT staff',
    8: 'Laborers',
    9: 'Low-skill Laborers',
    10: 'Managers',
    11: 'Medicine staff',
    12: 'Private service staff',
    13: 'Realty agents',
    14: 'Sales staff',
    15: 'Secretaries',
    16: 'Security staff',
    17: 'Waiters/barmen staff',
}


# Mapping for NAME_EDUCATION_TYPE
education = {'Secondary / secondary special' : 4, 'Higher education' : 1, 'Incomplete higher' : 2, 'Lower secondary' : 3, 'Academic degree' : 0}

# Mapping for Gender
Gender = {'M' : 1,'F' : 0, 'XNA' : 2}

Income = {'Working' : 5, 'State servant' : 3, 'Commercial associate' : 0, 'Student' : 4,
       'Pensioner' : 2, 'Maternity leave' : 1}

Reject_reason = {'XAP(X-Application Pending)' : 7, 'LIMIT(Credit Limit Exceeded)' : 2, 'SCO(Scope of Credit)' : 3,
                'HC(High Credit Risk)' : 1, 'VERIF(Verification Failed)' : 6, 'CLIENT(Client Request)' : 0, 
                'SCOFR(Scope of Credit for Rejection)' : 4, 'XNA(Not Applicable)' : 8, 'SYSTEM(System Error)' : 5}

status = {'Approved' : 0, 'Canceled' : 1, 'Refused' : 2, 'Unused offer' : 3}

Yield = {'low_normal' :3, 'middle' :4, 'XNA' :0, 'high' :1, 'low_action' :2}

with st.sidebar:
    st.image("https://static.vecteezy.com/system/resources/previews/010/518/840/original/digital-finance-and-banking-service-in-futuristic-background-bank-building-with-online-payment-transaction-secure-money-and-financial-innovation-technology-vector.jpg")
    opt = option_menu("Menu",
                    ["Home",'Matrix Insights','EDA','Model Prediction','Chat - GenAI',"About"],
                    icons=["house","table","bar-chart-line","graph-up-arrow","search", "exclamation-circle"],
                    menu_icon="cast",
                    default_index=0,
                    styles={"icon": {"color": "red", "font-size": "20px"},
                            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "-2px", "--hover-color": "blue"},
                            "nav-link-selected": {"background-color": "blue"}})
    
if opt=="Home":
    
        col,coll = st.columns([1,4],gap="small")
        with col:
            st.write(" ")
        with coll:
            st.markdown("# Bank Risk Controller System")
          
            st.write(" ")     
        st.markdown("### :red[*OVERVIEW*]")
        st.markdown("### *The expected outcome of this project is a robust predictive model that can accurately identify customers who are likely to default on their loans. This will enable the financial institution to proactively manage their credit portfolio, implement targeted interventions, and ultimately reduce the risk of loan defaults.*")
        col1,col2=st.columns([3,2],gap="large")
        with col1:
            st.markdown("### :red[*DOMAIN*] - Banking")
            st.markdown("""
                        ### :red[*TECHNOLOGIES USED*]     

                        ### *Python*
                        ### *Data Preprocessing*
                        ### *EDA(Exploratory Data Analysis)*
                        ### *Pandas*
                        ### *Numpy*
                        ### *Visualization*
                        ### *Machine Learning - Classification Model*
                        ### *Streamlit GUI*
                        
                        """)
        with col2:
                st.write(" ")
    
if opt=="Matrix Insights":
                #st.header(":red[Data Used]")
                #st.dataframe(df)

                st.header(":red[Model Performance]")
                data = {
                            "Algorithm": ["Decision Tree","Random Forest","KNN","XGradientBoost"],
                            "Accuracy": [86,86,88,98],
                            "Precision": [86,86,86,98],
                            "Recall": [86,86,86,98],
                            "F1 Score": [86,86,86,98]
                            
                            }
                dff = pd.DataFrame(data)
                st.dataframe(dff)
                st.markdown(f"## The Selected Algorithm is :red[*XGradient Boosting*] and its Accuracy is   :red[*98%*]")


if opt=="Model Prediction":
            
    # Streamlit form for user inputs
    st.markdown(f'## :red[*Predicting Customers Default on Loans*]')
    st.write(" ")
    
    with st.form("my_form"):
        col1, col2 = st.columns([5, 5])
        
        
        with col1:

            OCCUPATION_TYPE = st.selectbox("OCCUPATION TYPE", sorted(occupation.items()), format_func=lambda x: x[1], key='OCCUPATION_TYPE')[0]
            EDUCATION_TYPE = st.selectbox("EDUCATION TYPE", list(education.keys()), key='EDUCATION_TYPE')
            NAME_INCOME_TYPE = st.selectbox("INCOME TYPE",list(Income.keys()), key='NAME_INCOME_TYPE')
            TOTAL_INCOME = st.text_input("TOTAL INCOME PA", key='TOTAL_INCOME')
            CODE_REJECT_REASON = st.selectbox("CODE REJECTION REASON",list(Reject_reason.keys()), key='CODE_REJECT_REASON')
            NAME_CONTRACT_STATUS = st.selectbox("CONTRACT STATUS",list(status.keys()), key='NAME_CONTRACT_STATUS')
            

        with col2:
        
            NAME_YIELD_GROUP = st.selectbox("YIELD GROUP",list(Yield.keys()), key='NAME_YIELD_GROUP')
            CODE_GENDER = st.selectbox("CODE GENDER", list(Gender.keys()), key='CODE_GENDER')
            AGE = st.text_input("AGE", key="AGE")
            CLIENT_RATING = st.text_input("CLIENT RATING", key="CLIENT_RATING")
            DAYS_LAST_PHONE_CHANGE = st.text_input("PHONE CHANGE", key="DAYS_LAST_PHONE_CHANGE")
            DAYS_ID_PUBLISH = st.text_input("DAYS ID PUBLISH", key="DAYS_ID_PUBLISH")
            DAYS_REGISTRATION = st.text_input("DAYS REGISTRATION", key="DAYS_REGISTRATION")       
            
        submit_button = st.form_submit_button(label="PREDICT STATUS")

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #FBCEB1;
            color: red;
            width: 50%;
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)
  
    # Validate input
    flag = 0 
    pattern = r"^(?:\d+|\d*\.\d+)$"

    for i in [TOTAL_INCOME,AGE,CLIENT_RATING,DAYS_LAST_PHONE_CHANGE,DAYS_ID_PUBLISH,DAYS_REGISTRATION]:             
        if re.match(pattern, i):
            pass
        else:                    
            flag = 1  
            break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("Please enter a valid number, space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)  
    

    if submit_button and flag == 0:
        
        try:
            # Encode categorical variables
            le = LabelEncoder()

            Occupation = le.fit_transform([occupation[OCCUPATION_TYPE]])[0]
            Education = le.fit_transform([education[EDUCATION_TYPE]])[0]
            Income_type = le.fit_transform([Income[NAME_INCOME_TYPE]])[0]
            Reason = le.fit_transform([Reject_reason[CODE_REJECT_REASON]])[0]
            Status = le.fit_transform([status[NAME_CONTRACT_STATUS]])[0]
            yield_group = le.fit_transform([Yield[NAME_YIELD_GROUP]])[0]
            Genders = le.fit_transform([Gender[CODE_GENDER]])[0]
            #Occupation = occupation[OCCUPATION_TYPE]
            #Education = education[EDUCATION_TYPE]
            #Income_type = Income[NAME_INCOME_TYPE]
            Income_amt = int(TOTAL_INCOME.strip())
            #Reason = Reject_reason[CODE_REJECT_REASON]
            #Status = status[NAME_CONTRACT_STATUS]
            #yield_group = Yield[NAME_YIELD_GROUP]
            #Genders = Gender[CODE_GENDER]
            Age  = int(AGE.strip())
            Rating = int(CLIENT_RATING.strip())
            Phone  = int(DAYS_LAST_PHONE_CHANGE.strip())
            ID_Published = int(DAYS_ID_PUBLISH.strip())
            Registration = int(DAYS_REGISTRATION.strip())

            # Create sample array with encoded categorical variables
            sample = np.array([
                [
                    Occupation,
                    Education,
                    Income_type,
                    Income_amt,
                    Reason,
                    Status,
                    yield_group,
                    Genders,
                    Age,
                    Rating,
                    log_trans(Phone), 
                    log_trans(ID_Published), 
                    log_trans(Registration), 
                ]
            ])
                
            with open(r"xgbmodel_1.pkl", 'rb') as file:
                knn = pickle.load(file)

            #sample = scaler_loaded.transform(sample)
            pred = knn.predict(sample)

            if pred == 1:
                st.markdown(f' ## The status is: :red[Won\'t Repay]')
            else:
                st.write(f' ## The status is: :red[Repay]')
        except ValueError as e:
            st.error(f"Error processing inputs: {e}")
            st.write("Please check your input values. Only numeric values are allowed.")

if opt=="EDA":
    

    
    st.subheader(":red[Insights of Bank Risk Controller System]")

    col1,col2,col3 = st.columns(3)

             
    # detecting the skewed columns using plot
    plt.rcParams['figure.figsize'] = (4, 3)  # Set default figure size

    def skewplot(df, column):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(5, 4))  # Set figure size
        sns.boxplot(df[column])
        #sns.violinplot(df[column])
        plt.tight_layout()  # Adjust layout
        st.pyplot(use_container_width=True)
        

    with col1:
         
        skewed_columns= ['OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE','CODE_REJECT_REASON', 'NAME_CONTRACT_STATUS',
       'NAME_INCOME_TYPE', 'NAME_YIELD_GROUP', 'CODE_GENDER', 'TARGET', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH',
       'DAYS_LAST_PHONE_CHANGE', 'REGION_RATING_CLIENT_W_CITY', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION']

        for column in skewed_columns:
            st.write("### Skewed Data")
            skewplot(df, column)
                        
    with col2:

        # Apply square root transformation to the specified columns

        df['NAME_EDUCATION_TYPE_log'] = np.log(df['NAME_EDUCATION_TYPE'])
        df["PRODUCT_COMBINATION_log"] = np.log(df["PRODUCT_COMBINATION"])
        df["CODE_REJECT_REASON_log"] = np.log(df["CODE_REJECT_REASON"])
        df["NAME_INCOME_TYPE_log"] = np.log(df["NAME_INCOME_TYPE"])
        df["NAME_GOODS_CATEGORY_log"] = np.log(df["NAME_GOODS_CATEGORY"])
        df["AMT_INCOME_TOTAL_log"] = np.log(df["AMT_INCOME_TOTAL"])
        df["DAYS_LAST_PHONE_CHANGE_log"] = np.log(df["DAYS_LAST_PHONE_CHANGE"])
        df["DAYS_ID_PUBLISH_log"] = np.log(df["DAYS_ID_PUBLISH"])
        df["DAYS_REGISTRATION_log"] = np.log(df["DAYS_REGISTRATION"])


        
        skwed_columns_2=['NAME_EDUCATION_TYPE_log','CODE_REJECT_REASON_log',
              'NAME_INCOME_TYPE_log','AMT_INCOME_TOTAL_log',
              'DAYS_LAST_PHONE_CHANGE_log','DAYS_ID_PUBLISH_log','DAYS_REGISTRATION_log']
        for i in skwed_columns_2:
            st.write("### After Log Transformation")
            skewplot(df,i)

    with col3:


        def outlier(df,column):
            q1= df[column].quantile(0.25)
            q3= df[column].quantile(0.75)

            iqr= q3-q1

            upper_threshold= q3 + (1.5*iqr)
            lower_threshold= q1 - (1.5*iqr)

            df[column]= df[column].clip(lower_threshold, upper_threshold)


        outlier_columns= ['CODE_REJECT_REASON','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','DAYS_REGISTRATION']
        for i in outlier_columns:
            outlier(df,i)


        for i in outlier_columns:
            st.write("### After Interquartile Range (IQR)")
            skewplot(df,i)

if opt == "Chat - GenAI":        
       # Setup LangChain model and related components

# Load documents
    dir_files = DirectoryLoader("C:/Users/santh/MyFolder/", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = dir_files.load()

# Split documents into smaller chunks
    splited_text = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=50)
    documents_splitted = splited_text.split_documents(docs)

# Initialize embeddings and FAISS
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vect = FAISS.from_documents(documents_splitted, embedding)
    vect.save_local("db/faiss")

# Setup LLM model
    model_file = "C:/Users/santh/MyFolder/llama-2-7b-chat.ggmlv3.q8_0.bin"  # Replace with your model path
    llm_model = CTransformers(
    model=model_file,
    model_type="llama",
    config={"max_new_tokens": 500, "temperature": 0.01}
)

# Setup prompt template for the model
    prompt = """
    use the following information and answer the question based on that /
    provide only useful information /
    if you do not know, do not try to make wrong answers /
    if you do not know, just respond as 'I do not know' /

    show only the answer alone /
    Example: Can the Card be used immediately after it is purchased
    output: Yes, your State Bank Vishwa Yatra Foreign Travel Card can be used immediately after purchase except in India, Nepal and Bhutan.

    Context: {context}
    Questions: {question}
    """

    # Function to setup the retrieval QA model
    def sqa_prompt():
        prompt_temp = PromptTemplate(template=prompt, input_variables=['context', 'question'])
        return prompt_temp

    def ret_answer(llm_model_name, prompt_name, v_db):
        ques_ans = RetrievalQA.from_chain_type(
        llm=llm_model_name,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_name},
        retriever=v_db.as_retriever(search_kwargs={'k': 2})
    )
        return ques_ans

    def setup_mode():
        embb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        vdb = FAISS.load_local("db/faiss", embb, allow_dangerous_deserialization=True)
        pmt = sqa_prompt()
        get_answer = ret_answer(llm_model, pmt, vdb)
        return get_answer

# Initialize the QA model
    llm_qa_model = setup_mode()

# Streamlit app interface
    st.title("Bank FAQ Chatbot")
    st.write("Welcome to the FAQ Chatbot. Ask your questions related to bank services.")

# Input for user query
    asked_question = st.text_input("Ask a question:")

    if asked_question:
    # Get the answer from the model
        answer = llm_qa_model({'query': asked_question})
        st.write("Answer: ", answer)
    else:
        st.write("Please enter a question to get an answer.")


if opt=="About":
        
          
        st.markdown(f"### :red[ABOUT]")
        st.write(" ")
        st.markdown(f"#### ***In the financial industry, assessing the risk of customer default is crucial for maintaining a healthy credit portfolio. Default occurs when a borrower fails to meet the legal obligations of a loan. Accurate prediction of default can help financial institutions mitigate risks, allocate resources efficiently, and develop strategies to manage potentially delinquent accounts. This project aims to develop a predictive model to determine the likelihood of customer default using historical data.***")
        st.markdown(f"### :red[DATA DESCRIPTION]")
        st.markdown(f"#### ***The dataset provided contains multiple features that may influence the likelihood of a customer defaulting on a loan. These features include:*** ")
        st.write(" ")
        st.write(" ")
        
        st.markdown(f"#### :red[***Personal Information:***] Age, gender, etc.")
        st.markdown(f"#### :red[***Credit History:***] Previous loan defaults, credit score, number of open credit lines, etc.")
        st.markdown(f"#### :red[***Financial Status:***] Annual income, current debt, loan amount, etc.")
        st.markdown(f"#### :red[***Employment Details:***] Employment status, etc. The target variable 0: No default 1: Default")

        st.write(" ")
        st.write(" ")

        st.markdown(f"### :red[CONCLUSION]")
        st.write(" ")
        st.markdown(f"#### ***The expected outcome of this project is a robust predictive model that can accurately identify customers who are likely to default on their loans. This will enable the financial institution to proactively manage their credit portfolio, implement targeted interventions, and ultimately reduce the risk of loan defaults.***")


        co,coo=st.columns([3,7],gap="small")
        with coo:
           st.markdown(f"#### *Feel free to delve into the project on my GitHub repository* ")
        with co:
             st.write(" ")
             





            


    








