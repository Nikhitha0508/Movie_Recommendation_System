#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import pandas as pd


# In[84]:


movies_data = pd.read_csv('tmdb_5000_movies.csv')
credits_data = pd.read_csv('tmdb_5000_credits.csv')


# In[85]:


movies_data.head()


# In[86]:


credits_data.head()['cast'].values


# In[87]:


movies_data.shape


# In[88]:


credits_data.shape


# In[90]:


movies= movies_data.merge(credits_data,on="title")


# In[91]:


movies.head()


# In[92]:


#genres
#id
#keywords
#title
#overview
#cast
#crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[93]:


movies.info()


# In[94]:


movies.head()


# In[95]:


movies.isnull().sum()


# In[96]:


movies.dropna(inplace = True)


# In[97]:


movies.duplicated().sum


# In[98]:


movies.iloc[0].genres


# In[99]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'


# In[100]:


import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[101]:


movies['genres']= movies['genres'].apply(convert)


# In[102]:


movies.head()


# In[103]:


movies['keywords'].apply(convert)


# In[104]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[105]:


movies.head()


# In[106]:


movies['cast'][0]


# In[107]:


def convert3(obj):
    L = []
    counter =0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter = counter+1
        else:
            break 
    return L


# In[108]:


movies['cast']=movies['cast'].apply(convert)


# In[109]:


movies.head()


# In[110]:


movies['crew'][0]


# In[111]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L 


# In[112]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[113]:


movies.head()


# In[114]:


movies['overview'][0]


# In[115]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[116]:


movies.head()


# In[117]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[119]:


movies.head()


# In[120]:


movies['tags'] = movies['overview'] + movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[121]:


movies.head()


# In[122]:


new_df = movies[['movie_id','title','tags']]


# In[123]:


new_df


# In[124]:


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))


# In[125]:


new_df['tags'][0]


# In[126]:


new_df['tags'].apply(lambda x:x.lower())


# In[127]:


new_df.head()


# In[128]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[129]:


cv.fit_transform(new_df['tags']).toarray()


# In[130]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[131]:


vectors= cv.fit_transform(new_df['tags']).toarray()


# In[132]:


vectors


# In[133]:


vectors[0]


# In[134]:


cv.get_feature_names()


# In[135]:


get_ipython().system(' pip install nltk')


# In[136]:


import nltk


# In[137]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[138]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)    


# In[139]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[140]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[141]:


cv.fit_transform(new_df['tags']).toarray()


# In[142]:


vectors[0] 


# In[143]:


cv.get_feature_names()


# In[144]:


from sklearn.metrics.pairwise import cosine_similarity


# In[145]:


similarity= cosine_similarity(vectors)


# In[146]:


sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x:x[1])[1:6]


# In[148]:


similarity.shape


# In[149]:


similarity[1]


# In[150]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True,key = lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    


# In[151]:


recommend('Batman Begins')


# In[152]:


import pickle


# In[153]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[154]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




