#!/usr/bin/env python
# coding: utf-8

# In[9]:



from pymol import cmd
from drugpy.ftmap.core import load_atlas
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import seaborn as sb
import statistics
from pharmacophore import InteractionKind, Feature, PharmacophoreJsonWriter


get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


caminhos = ('/home/gessualdo/Downloads/box_pdb/results_atlas_gessualdo/3eml_box_v2.pdb',
            '/home/gessualdo/Downloads/fragmap/hotspots_3eml.pse', 
            '/home/gessualdo/Documentos/maps_fragmap/3EML' )


# In[18]:


def get_coord_of_hotspot_from_ftmap(caminho_ftmap:str):

    ftmap = load_atlas(caminho_ftmap, plot = False)

    objeto = caminho_ftmap.split("/")[-1].split('.')[-2]
   
    hotspots, clusters  = ftmap[objeto]

    max_s = 0
    for hotspot in hotspots:
        if hotspot.strength > max_s:
            hotspot_max = hotspot
            max_s = hotspot.strength

    coorOfHotspot = cmd.get_extent(hotspot_max.selection)
    
    return coorOfHotspot


# In[22]:


#Criar contorno em torno do hot spot druggable.


def get_features_from_ftmap_and_fragmap(caminho_ftmap:str,
                                        caminho_fragmap:str, 
                                        caminho_dump:str, 
                                        tipo:str, 
                                        level_param:int,
                                        radius_mult: float
                                        ):

    coorOfHotspot = get_coord_of_hotspot_from_ftmap(caminho_ftmap)

    x_min = coorOfHotspot[0][0]
    x_max = coorOfHotspot[1][0]
    y_min = coorOfHotspot[0][1]
    y_max = coorOfHotspot[1][1]
    z_min = coorOfHotspot[0][2]
    z_max = coorOfHotspot[1][2]


    cmd.load(caminho_fragmap)
    cmd.dump(f'{caminho_dump}/{tipo}.txt', tipo)

    level = []
    x_coor = []
    y_coor = []
    z_coor = []
    coordenadas = []

    with open(f'{caminho_dump}/{tipo}.txt', 'r') as r:
        for linha in r:
            linha = linha.split()
           
            level = float(linha[3])
            x = float(linha[0])
            y = float(linha[1])
            z = float(linha[2])

            if x >  x_min and x < x_max and y > y_min and y <  y_max and z > z_min and z < z_max and                     level >= level_param :
                
                coordenadas.append((float(level), float(x), float(y), float(z)))

        
    get_key = lambda elem: elem[0]
    
    coordenadas.sort(key = get_key, reverse = True)
    
    coor_ = coordenadas

    #Melhor score para o clusterização

    range_n_clusters = range(2,8)

    X = [(x, y, z) for _, x, y, z in coor_] 

    score = []

    for n_clusters in range_n_clusters:

        clusters = KMeans(n_clusters = n_clusters, random_state= 0)
        cluster_labels = clusters.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        
        score.append(silhouette_avg)
       


    #utilizar o melhor score para determinar o valor de k -> métrica silhouette score. 
    #quanto > o valor do score, melhor a divisão de clusters.

    centroides = []    
    for ncluster, best_score in zip(range_n_clusters, score):
        if best_score == max(score):
            kmeans = KMeans(n_clusters = ncluster, random_state=0).fit(X)
            centerofmass = kmeans.cluster_centers_
            centroides = centerofmass
            labels = kmeans.labels_ 
            
    
            
            df = pd.DataFrame(X)
            df['labels'] = labels
            df['contorno'] = [level for level, _, _, _ in coor_]
            df = df.rename(columns={0:'x_coor', 1:'y_coor', 2:'z_coor'})
     
    feats = {} 
    radius = []
    points_and_radius = []
    
    for k in range(0, len(centroides)):
       
        #selecionar os 2 primeiros clusters de maior contorno 
        max_contours = df.groupby('labels')['contorno'].sum()
        
        df_k = df[df['labels'] == k]  
        
        distance_ = []
        for row in df_k.iterrows():
            
            xyz = list(row[1][0:3])
            
            
            distance_.append(distance.euclidean(xyz, centroides[k]))

        points_and_radius.append((centroides[k], statistics.mean(distance_) * radius_mult))        
        
    return points_and_radius, max_contours


# In[16]:


get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'acceptor', 17, 1)
  


# In[20]:


def build_pharmacophore(arquivo_saida:str, caminhos:tuple):
    
    points_and_radius_acceptor, _ = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'acceptor', 17, 1)
    points_and_radius_donor, _ = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'donor', 17, 1)
    points_and_radius_apolar, _ = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'apolar', 17, 0.5)
    
    pharmacophore_writer = PharmacophoreJsonWriter()
    
    feats = []
    for (x ,y, z), radius in points_and_radius_acceptor:
        feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, radius))    
    
    for (x ,y, z), radius in points_and_radius_donor:
        feats.append(Feature(InteractionKind.DONOR, x, y, z, radius))  
    
    for (x ,y, z), radius in points_and_radius_apolar:
        feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, radius))

    pharmacophore_writer.write(feats, arquivo_saida)
    
    


# In[21]:


build_pharmacophore(f'{caminhos[2]}/feat.json', caminhos)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[96]:





# In[ ]:





# In[ ]:




