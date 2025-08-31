# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_experimental.sql import SQLDatabaseChain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# from langchain.utilities import SQLDatabase
# from langchain.vectorstores import FAISS
# from langchain.prompts import SemanticSimilarityExampleSelector
# from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
# from langchain.prompts.prompt import PromptTemplate
# from langchain.prompts import FewShotPromptTemplate
# from few_shots import few_shots
# load_dotenv()
# def get_few_shot_db_chain():
    
#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.6)
#     db_user="root"
#     db_password='root'
#     db_host="localhost"
#     db_name="atliq_tshirts"

#     db=SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)
    
#     embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     vectorstore_texts = [
#     " ".join(str(v) for v in example.values()) for example in few_shots
#     ]
#     vector_store = FAISS.from_texts(vectorstore_texts, embedding=embeddings,metadatas=few_shots)
#     example_selector=SemanticSimilarityExampleSelector(
#     vectorstore=vector_store,
#     k=2
#     )

#     example_prompt = PromptTemplate(
#     input_variables=["Question", "SQLQuery", "SQLResult","Answer"],
#     template="--\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}" 
#     )
#     few_shot_template=FewShotPromptTemplate(
#     example_selector=example_selector,
#     example_prompt=example_prompt,
#     prefix=_mysql_prompt,
#     suffix=PROMPT_SUFFIX,
#     input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix

#     )

#     chain=SQLDatabaseChain.from_llm(llm,db,verbose=False, prompt=few_shot_template)
#     return chain






# if __name__ == "__main__":
#     chain = get_few_shot_db_chain()
#     print(chain.run("How many White color tshirt left"))


# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_experimental.sql import SQLDatabaseChain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# from langchain.utilities import SQLDatabase
# from langchain.vectorstores import FAISS
# from langchain.prompts import SemanticSimilarityExampleSelector
# from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
# from langchain.prompts.prompt import PromptTemplate
# from langchain.prompts import FewShotPromptTemplate
# from few_shots import few_shots
# load_dotenv()
# def get_few_shot_db_chain():
    
#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.6)
#     db_user="root"
#     db_password='root'
#     db_host="localhost"
#     db_name="atliq_tshirts"

#     db=SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)
    
#     embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     vectorstore_texts = [
#     " ".join(str(v) for v in example.values()) for example in few_shots
#     ]
#     vector_store = FAISS.from_texts(vectorstore_texts, embedding=embeddings,metadatas=few_shots)
#     example_selector=SemanticSimilarityExampleSelector(
#     vectorstore=vector_store,
#     k=2
#     )
#     # Corrected example_prompt
#     example_prompt = PromptTemplate(
#     input_variables=["Question", "SQLQuery", "SQLResult","Answer"],
#     template="--\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}" 
#     )

#     few_shot_template=FewShotPromptTemplate(
#     example_selector=example_selector,
#     example_prompt=example_prompt,
#     prefix=_mysql_prompt,
#     suffix=PROMPT_SUFFIX,
#     input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
#     )

#      # Use a custom chain to intercept and clean the query
#     class CustomSQLDatabaseChain(SQLDatabaseChain):
#         def _call(self, inputs):
#             llm_output = super()._call(inputs)
#             # Check if the output is a string and contains the unwanted prefix
#             if isinstance(llm_output, str) and llm_output.strip().startswith('SQLQuery:'):
#                 llm_output = llm_output.strip().replace('SQLQuery:', '').strip()
#             return llm_output
    
#     chain = CustomSQLDatabaseChain.from_llm(llm, db, verbose=False, prompt=few_shot_template)
#     return chain

# if __name__ == "__main__":
#     chain = get_few_shot_db_chain()
#     print(chain.run("How many White color tshirt left"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.utilities import SQLDatabase
from langchain.vectorstores import FAISS
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from few_shots import few_shots
load_dotenv()

def get_few_shot_db_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.6)
    db_user="root"
    db_password='root'
    db_host="localhost"
    db_name="atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore_texts = [" ".join(str(v) for v in example.values()) for example in few_shots]
    vector_store = FAISS.from_texts(vectorstore_texts, embedding=embeddings, metadatas=few_shots)
    
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vector_store,
        k=2
    )
    
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
    )

     # Use a modified prompt suffix to explicitly instruct the model on the output format
    PROMPT_SUFFIX_CLEAN = """
    You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run.
    The query must be a single line of SQL without any extra text, labels, or prefixes like 'SQLQuery:'.
    
    {table_info}
    {few_shot_examples}
    Question: {input}
    SQLQuery:
    """

    few_shot_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],
    )

    # Use a custom chain to intercept and clean the query
    class RobustSQLDatabaseChain(SQLDatabaseChain):
        def _call(self, inputs):
            llm_output = super()._call(inputs)
            
            # Aggressive cleaning of the LLM's output
            if isinstance(llm_output, str):
                # Remove common prefixes
                cleaned_output = llm_output.strip().lower()
                for prefix in ['sqlquery:', '```sql', '```','SQLQuery:\n','SQLQuery:']:
                    if cleaned_output.startswith(prefix):
                        llm_output = llm_output.strip().replace(prefix, '', 1).strip()
                
                # Remove any trailing markdown block delimiters
                for sufix in[';','```','--']:
                    if llm_output.strip().endswith(sufix):
                        llm_output = llm_output.strip().rstrip(sufix).strip()
                    
                # The final output should be just the SQL query
                return llm_output
            
            return llm_output
    
    chain = RobustSQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_template)
    return chain

if __name__ == "__main__":
    chain = get_few_shot_db_chain()
    print(chain.run("how many white color t shirts left")) 