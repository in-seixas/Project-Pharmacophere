#!/usr/bin/env python
# coding: utf-8



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





def get_coord_of_hotspot_from_ftmap(caminho_ftmap:str):

    ftmap = load_atlas(caminho_ftmap, plot = False, table = False)

    objeto = caminho_ftmap.split("/")[-1].split('.')[-2]
   
    hotspots, clusters  = ftmap[objeto]

    max_s = 0
    for hotspot in hotspots:
        if hotspot.strength > max_s:
            hotspot_max = hotspot
            max_s = hotspot.strength

    coorOfHotspot = cmd.get_extent(hotspot_max.selection)
    
    return coorOfHotspot, hotspot_max.selection


# In[53]:


#Criar contorno em torno do hot spot druggable.


def get_features_from_ftmap_and_fragmap(caminho_ftmap:str,
                                        caminho_fragmap:str, 
                                        caminho_dump:str, 
                                        tipo:str, 
                                        level_param:int,
                                        radius_mult: float
                                        ):

    coorOfHotspot, hotspot_sel = get_coord_of_hotspot_from_ftmap(caminho_ftmap)

    x_min = coorOfHotspot[0][0]
    x_max = coorOfHotspot[1][0]
    y_min = coorOfHotspot[0][1]
    y_max = coorOfHotspot[1][1]
    z_min = coorOfHotspot[0][2]
    z_max = coorOfHotspot[1][2]
    
    
    cmd.load(caminho_fragmap, partial = 1)
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
                
                level2 = 1 + cmd.count_atoms(f'x > {x-3} and x < {x + 3}' 
                            f' and y > {y-3} and y < {y+3} and z > {z-3} and z < {z + 3} and {hotspot_sel}')
                
                coordenadas.append((float(level)*level2, float(x), float(y), float(z)))

        
    get_key = lambda elem: elem[0]
    
    coordenadas.sort(key = get_key, reverse = True)
    
    coor_ = coordenadas

    #Melhor score para o clusterização

    range_n_clusters = range(2,7)

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
        max_contours = df.groupby(['labels'])['contorno'].sum()
        
        df_k = df[df['labels'] == k]  
        
        distance_ = []
        for row in df_k.iterrows():
            
            xyz = list(row[1][0:3])
            
            
            distance_.append(distance.euclidean(xyz, centroides[k]))
            
            
            

        points_and_radius.append((centroides[k], statistics.mean(distance_) * radius_mult))        
        
       
        
    return points_and_radius




def build_pharmacophore(arquivo_saida:str, caminhos:tuple):
    
    points_and_radius_acceptor, _ = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'acceptor', 17, 1)
    points_and_radius_donor, _ = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'donor', 17, 1)
    points_and_radius_apolar, _ = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'apolar', 17, 1)
    
    pharmacophore_writer = PharmacophoreJsonWriter()
    
    feats = []
    for (x ,y, z), radius in points_and_radius_acceptor:
        
        if radius > 1.5:
            
            feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, 1.5)) 
       
        elif radius < 1.0:
            
            feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, 1.0))
          
        else:
            
            feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, radius)) 
    
    for (x ,y, z), radius in points_and_radius_donor:
        
        if radius > 1.5:
            
            feats.append(Feature(InteractionKind.DONOR, x, y, z, 1.5))  
            
        elif radius < 1.0:
            
            feats.append(Feature(InteractionKind.DONOR, x, y, z, 1.0)) 
            
        else:
            
             feats.append(Feature(InteractionKind.DONOR, x, y, z, radius)) 
              
    for (x ,y, z), radius in points_and_radius_apolar:
        
        if radius > 1.5:
            
            feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, 1.5))
            
        elif radius < 1.0:
            
            feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, 1.0))     
        
        else:
            feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, radius))

    pharmacophore_writer.write(feats, arquivo_saida)
    
    







#build_pharmacophore(f'{caminhos[2]}/feat.json', caminhos)




caminhos = [
    "/home/gessualdosoj/Downloads/pharmacophore-20211023T012057Z-001/pharmacophore/2azr_box_lig.pdb",
    "/home/gessualdosoj/Downloads/hotspots_2AZR.pse",
    "/home/gessualdosoj/Downloads/aa",
    "/home/gessualdosoj/Downloads/"]






def write_query_to_sybyl(caminhos:tuple):
    
    coordenadas_donor = get_features_from_ftmap_and_fragmap(caminhos[0], 
                                    caminhos[1], 
                                    caminhos[2], 
                                    "donor", 17, 1)
    
    coordenadas_apolar = get_features_from_ftmap_and_fragmap(caminhos[0], 
                                    caminhos[1], 
                                    caminhos[2], 
                                    "apolar", 17, 1)
    
    coordenadas_acceptor = get_features_from_ftmap_and_fragmap(caminhos[0], 
                                    caminhos[1], 
                                    caminhos[2], 
                                    "acceptor", 17, 1)
    
    
    sum_lines = (len(coordenadas_donor) + len(coordenadas_apolar) + len(coordenadas_acceptor)) *2

    coords_donor = []
    radius_donor = []

    for i in coordenadas_donor:

        coords_donor.append(i[0])
        radius_donor.append(i[1])

    coords_apolar = []
    radius_apolar = []


    for i in coordenadas_apolar:
        coords_apolar.append(i[0])
        radius_apolar.append(i[1])

    coords_acceptor = []
    radius_acceptor = []


    for i in coordenadas_acceptor:
        coords_acceptor.append(i[0])
        radius_acceptor.append(i[1])

    
    with open(caminhos[3] + "query.mol2", "w") as query:
        
        
        query.write(f"""

@<TRIPOS>MOLECULE
Complex_query
    1     0     0    {sum_lines}     0
SMALL
No_CHARGES


@<TRIPOS>ATOM2
      1 DU       -1000.0000 -1000.0000 -1000.0000 Du        0 <0>         0.0000 
@<TRIPOS>NORMAL
@<TRIPOS>U_FEAT """)
        

        for i, (c, r) in enumerate(zip(coords_donor, radius_donor)):
            
            x, y, z = c
            raio = r

            query.write(f""" 
4 13 DONOR_ATOM{i} DONOR_ATOM {x} {y} {z} 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 14 
5 14 SPATIALPT{i} {raio} {x} {y} {z} 0 1 DONOR_ATOM{i} BLUE""")


        for i, (c, r) in enumerate(zip(coords_apolar, radius_apolar)):

            x, y, z = c
            raio = r

            query.write(f""" 
4 13 APOLAR_ATOM{i} HYDROPHOBIC {x} {y} {z} 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 14 
5 14 SPATIALPT{i} {raio} {x} {y} {z} 0 1 APOLAR_ATOM{i} YELLOW""")


        for i, (c, r) in enumerate(zip(coords_acceptor, radius_acceptor)):

            x, y, z = c
            raio = r

            query.write(f""" 
4 13 ACCEPTOR_ATOM{i} ACCEPTOR_ATOM {x} {y} {z} 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 14 
5 14 SPATIALPT{i} {raio} {x} {y} {z} 0 1 ACCEPTOR_ATOM{i} RED""")







