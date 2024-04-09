# Movie-Recommendation-System

## Here we will make a program to input a movie name and recommend the user movies that are similar to the inputted movie name. The program will be deployed later

1. ## Loading of Data
   Necessary libraries are loaded and the dataset is taken from kaggle

   ```

   # important libraries are imported
   import numpy as np
   import pandas as pd


   # The information regrading the movies are obrained from two datasets, namely credits and movies. Both csv files are uploaded 
   to      dataframes

   credits=pd.read_csv(r'D:\PROJECTS\Movie-Recommendation-System\dataset\tmdb_5000_credits.csv')
   movies= pd.read_csv(r'D:\PROJECTS\Movie-Recommendation-System\dataset\tmdb_5000_movies.csv')  




   ```

   ### Both dataframes are merged based on column named 'id' in movies and movie_id in credits dataframe

   ```
   merged_df=movies.merge(credits, left_on='id', right_on='movie_id')

   ```


2. ## Feature Selection

   Out of all the 24 features available we are taking only 7 features for recommendation.

   ```
   # choosing relevant features for the model
   #genre
   #id  - 
   #keywords
   # overview
   # original_title
   # cast
   #crew


    L=['original_title','genres','id','keywords','cast','crew','overview']

    sorted_df=merged_df[L]
    sorted_df.head(2)
      


    ```
 
  
  <img width="662" alt="image" src="https://github.com/sreehari32/movie_recommendation_system/assets/51872549/72a19640-a888-4066-8c79-305aeadb6e55">


3. ## Data Preprocessing

   The values in the columns are stored as dictionaries. These values are extracted using customised function and new columsn are 
   made for further easy processings
  
   '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science 
    Fiction"}]'
        

    We can see that the genres are stored as list of dictionaries. We need to define a function to extract the genres and store 
    it as a list


    ```
    import ast
    def extraction(obj):
    lis=[]    
    for i in ast.literal_eval(obj):
        lis.append(i['name'])
    return lis
    ```

    ```
    # we are extrating the genres from the dictionary and making it as a list
    sorted_df['genres']=sorted_df['genres'].apply(extraction)
    ```
 
   ###similar to genre, the keywords also needs to be extracted from the dictionary format

    ```
    sorted_df['keywords']=sorted_df['keywords'].apply(extraction)
    ```

   ### Similar to the two previous extractions, top 3 actors names are also extracted out
    
    ```
    #function to extract top 3 cast
    import ast
    def extraction_1(obj):
        lis=[]
        j=0
        for i in ast.literal_eval(obj):
            if j<3:
                lis.append(i['name'])
                j+=1
    return lis



    sorted_df['cast']=sorted_df['cast'].apply(extraction_1)


    ```

   ### Among the crew members only director name is extracted
   

    ```
    def fetch_director(obj):
    import ast
    lis=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            lis.append(i['name'])
            break
    return lis


    sorted_df['crew']=sorted_df['crew'].apply(fetch_director)

    ```


   ###  After this we  convert the texts in the overview column to list. This will help us in language processing.  

    ```
      sorted_df['overview']=sorted_df['overview'].apply(lambda x:x.split())


    ```

    ```
    # The spaces between the first name and second names 
    # of cast and director will complicate our language processing  .So wer are 
    #removing the spaces between the names of cast and directors
    for j in ['genres','keywords', 'cast','crew']:
        sorted_df[j]=sorted_df[j].apply(lambda x:[i.replace(" ","")  for i in x])

    ```

   ### All the values of genre, keywords, cast and crew and overview are concatenated to form a single list

    ```
    sorted_df['tag']= sorted_df['overview']+sorted_df['genres']+sorted_df['keywords']+sorted_df['cast']+sorted_df['crew']

    ```

   ### A new dataframe is formed where only columns like id, original, title, and tag is present

    ```
    new_df=sorted_df[['id','original_title','tag']]
    new_df.head()
    
    ```
  
  <img width="514" alt="image" src="https://github.com/sreehari32/movie_recommendation_system/assets/51872549/1754aa0b-40e2-40c1-a632-c55331871866">




   ### Converting lists in 'tag' to strings
  
    ```
    new_df['tag']=new_df['tag'].apply(lambda x: " ".join(x))

    new_df['tag'][0]

    ```

<img width="673" alt="image" src="https://github.com/sreehari32/movie_recommendation_system/assets/51872549/e7dc272a-00a5-41b4-92b6-892dc3018210">




3. ## Natural Language Processing
   nltk library is imported and PorterStemmer is used to find the root words

    ```
    import nltk
    from nltk.stem.porter import PorterStemmer 
    ps=PorterStemmer()

    def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

    new_df['tag']=new_df['tag'].apply(lambda x: x.lower())


    new_df['tag'].apply(stem)


    ```


<img width="368" alt="image" src="https://github.com/sreehari32/movie_recommendation_system/assets/51872549/f6f41a9b-3ea7-48cc-87fa-b958208e745a">



 ### Count Vectorisation

    using count vectorisation, words are converted to vectors
    
     ```
     from sklearn.feature_extraction.text import CountVectorizer
     cv=CountVectorizer(max_features=5000,stop_words='english')
     vectors=cv.fit_transform(new_df['tag']).toarray()

     ```

  ### Checking the similarity
      The proximity of the different vectors are measured using the cosine similarity function 

      ```
      
      from sklearn.metrics.pairwise import cosine_similarity
      similarity=cosine_similarity(vectors)
      print(similarity[0]) # vector distance of first vector with rest of the 4800

      ```

4. # Poster fetching
     A function is defined that will return the movie poster url when we pass the movie id. The images are extracted using TMDB  
     API 

      ```
      import requests

      def movie_poster_fetch(movie_id):

          url = "https://api.themoviedb.org/3/movie/{}?api_key=2bb614d1e9ddaef999e871eb20076d6a".format(
          movie_id )
          data = requests.get(url)
          data = data.json()
          poster_path = data["poster_path"]
    
          full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
          return full_path

      ```


5. # Movie Recommendation
     A function called recommend is defined which will take movie_name as parameter
     The index of the refered movie is identified.
     All the movies near to the this particular index is found and top 4 is selected.
      


      ```
      movies_list=[]
      poster_links_list=[]

      def recommend(movie_name):
    
          movie_index=new_df[new_df['original_title']== movie_name].index[0]
          distances=similarity[movie_index]
          movies_list= sorted(list(enumerate(distances)),reverse= True, key= lambda X: X[1])[1:5]
          for i in movies_list:
               poster_links_list.append(movie_poster_fetch(new_df.loc[i[0],'id']))   
    
           return movies_list, poster_links_list

      ```


















 









   
