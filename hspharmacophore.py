from numpy import core
from pymol import cmd
from drugpy.ftmap.core import load_atlas
from drugpy.commons import fo_
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import seaborn as sb
import statistics
from pharmacophore import InteractionKind, Feature, PharmacophoreJsonWriter



def get_active_site(caminho_ftmap:str, saida_active_site: str ):

    ftmap = load_atlas(caminho_ftmap, plot = False, table = False)

    objeto = list(ftmap.keys())[0]
    
    hotspots, clusters  = ftmap[objeto]

    all_hotspots = []
    join_hs_D = []
 
    for hotspot in hotspots:
        
        all_hotspots.append(
            {
            "Selection": hotspot.selection, 
            "Strenght": hotspot.strength,
            "Klass": hotspot.klass  })

    hs_D = list(filter(lambda x: x["Klass"] == "D", all_hotspots))
    
    hs_D.sort(key= lambda x : x["Strenght"], reverse= True)

    hotspot_max = list(hs_D[0].values())[0]   #hs mais forte

    for dict_hs in hs_D:
        
        fo = fo_(hotspot_max, dict_hs["Selection"])
        
        if fo >= 0.5: 
                
            join_hs_D.append(dict_hs["Selection"])

    
    all_druggable = " or ".join(join_hs_D)
      
    print("ALL DRUGGABLE", all_druggable) 

    cmd.create("all_druggable", all_druggable)
                
    cmd.select("active_site", "byres polymer within 8 of all_druggable")
    
    cmd.save(saida_active_site, "active_site", format="pdb")

    cmd.reinitialize()
        



def get_coord_of_hotspot(caminho_ftmap:str):

    ftmap = load_atlas(caminho_ftmap, plot = False, table = False)

    objeto = list(ftmap.keys())[0]
    
    hotspots, clusters  = ftmap[objeto]

    hotspots_druggables = []
    join_hs_D = []
 
    for hotspot in hotspots:
        
        hotspots_druggables.append(
            {
            
            "Selection": hotspot.selection, 
            "Strenght": hotspot.strength,
            "Klass": hotspot.klass  })

    hs_D = list(filter(lambda x: x["Klass"] == "D", hotspots_druggables))
    
    hs_D.sort(key= lambda x : x["Strenght"], reverse= True)

    hotspot_max = list(hs_D[0].values())[0]  #hs mais forte

    for dict_hs in hs_D:
        
        fo = fo_(hotspot_max, dict_hs["Selection"])
        
        if fo >= 0.5: 
                
            join_hs_D.append(dict_hs["Selection"])

        
    all_druggable = " or ".join(join_hs_D)
  
    print("ALL_DRUGGABLE", all_druggable)

    cmd.create("all_druggable", all_druggable)
                
    coorOfHotspot = cmd.get_extent("all_druggable")

    return coorOfHotspot,  "all_druggable"



#Criar contorno em torno do hot spot druggable.


def get_features_from_ftmap_and_fragmap(caminho_ftmap:str,
                                        caminho_fragmap:str, 
                                        caminho_dump:str, 
                                        tipo:str,
                                        chemical_class:str, 
                                        level_param:int,
                                        radius_mult: float
                                        ):

    coorOfHotspot, hotspot_sel  = get_coord_of_hotspot(caminho_ftmap)


    x_min = coorOfHotspot[0][0]
    x_max = coorOfHotspot[1][0]
    y_min = coorOfHotspot[0][1]
    y_max = coorOfHotspot[1][1]
    z_min = coorOfHotspot[0][2]
    z_max = coorOfHotspot[1][2]
    

    cmd.load(caminho_fragmap, partial = 1)
    cmd.dump(f'{caminho_dump}/{tipo}.txt', tipo)
    

    level = []
    coordenadas = []

    with open(f'{caminho_dump}/{tipo}.txt', 'r') as r:
        for linha in r:
            x, y, z, level = map(float, linha.split())

            if x >  x_min and x < x_max and y > y_min and y <  y_max and z > z_min and z < z_max and level >= level_param :
                
                dc = 1 + cmd.count_atoms(f'name {chemical_class} and  x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hotspot_sel}')
        
                coordenadas.append((level, dc, x, y, z))
    
    get_key = lambda elem: elem[0]
    
    coordenadas.sort(key = get_key, reverse = True)

    df = pd.DataFrame(coordenadas, columns=["Nível", "Densidade", "X", "Y", "Z"])


    densidade = df["Densidade"].quantile(0.75)

    df.query("Densidade >= @densidade", inplace=True)


    #Melhor score para o clusterização

    range_n_clusters = range(2,6)

    X = df.iloc[:, [2,3,4]]

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
            
    df.loc[:, "Labels"] = labels 

    df.to_csv(f"{caminho_dump}/matriz_{tipo}.csv")
            
    points_and_radius = []
    
    for k in range(0, len(centroides)):
       
        #selecionar os 2 primeiros clusters de maior contorno 
        max_contours = df.groupby(['Labels'])['Densidade'].sum()


        df_k = df.loc[df['Labels'] == k]  
     
        distance_ = []

       
        for row in df_k.iterrows():
            
            xyz = list(row[1][2:5])

            distance_.append(distance.euclidean(xyz, centroides[k]))


        points_and_radius.append((centroides[k], statistics.mean(distance_)*radius_mult, max_contours[k]))

 
        points_and_radius.sort(key= lambda x: x[2], reverse= True)


   
    return points_and_radius



def get_density_correlation(caminho_ftmap, caminho_saida, level):

    _, hs = get_coord_of_hotspot(caminho_ftmap)

    points_and_radius_acceptor = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "acceptor", "acc", level, 1)
    points_and_radius_donor = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "donor", "don*", level, 1)
    points_and_radius_apolar = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "apolar", "c*", level, 1)

   
    centers_acceptor = []
   
    for (x ,y, z), radius, _ in points_and_radius_acceptor:


        dca = cmd.count_atoms(f' acc. and x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hs}')


        centers_acceptor.append({
            "X, Y, Z":(x,y,z),
            "TIPO":"ACCEPTOR",
            "DC": dca

        })       

    centers_donor = []

    for (x ,y, z), radius, _ in points_and_radius_donor:

        dcd = cmd.count_atoms(f'don. and x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hs}')

        centers_donor.append({
            "X, Y, Z":(x,y,z),
            "TIPO":"DONOR",
            "DC": dcd
        })
    

    centers_apolar = []

    for (x ,y, z), radius, _ in points_and_radius_apolar:

        dch = cmd.count_atoms(f'c. and x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hs}')

        centers_apolar.append({
            "X, Y, Z":(x,y,z),
            "TIPO":"HYDROFOBIC",
            "DC": dch
        })

    centers_model = list(centers_donor + centers_acceptor + centers_apolar)

    centers_model.sort(key= lambda x : x["DC"], reverse= True)

    feats = []

    pharmacophore_writer = PharmacophoreJsonWriter()

    for dicts in centers_model:
        for key, values in dicts.items():

            if dicts[key] == "ACCEPTOR":
                
                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                
                feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, 1))
            
            elif dicts[key] == "DONOR":

                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]

            
                feats.append(Feature(InteractionKind.DONOR, x, y, z, 1))

            elif dicts[key] == "HYDROFOBIC":


                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]

                feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, 1))       

    pharmacophore_writer.write(feats, caminho_saida)




def build_pharmacophore(arquivo_saida:str, caminhos:tuple):
    
    points_and_radius_acceptor = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'acceptor', "acc", 10, 1)
    points_and_radius_donor = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'donor', "don", 10, 1)
    points_and_radius_apolar = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], 'apolar', "c", 10, 1)
    
    pharmacophore_writer = PharmacophoreJsonWriter()
    
    feats = []
    for (x ,y, z), radius, _ in points_and_radius_acceptor:

        if radius > 1.0:
    
            feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, 1.0)) 

        elif radius < 0.5:

            feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, 0.5))

        else:

            feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, radius))


    for (x ,y, z), radius, _ in points_and_radius_donor:

        if radius > 1.0:

            feats.append(Feature(InteractionKind.DONOR, x, y, z, 1.0))

        elif radius < 0.5:

            feats.append(Feature(InteractionKind.DONOR, x, y, z, 0.5))

        else:

            feats.append(Feature(InteractionKind.DONOR, x, y, z, radius)) 
        
    for (x ,y, z), radius, _ in points_and_radius_apolar:

        if radius > 1.2:

            feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, 1.5)) 

        elif radius < 1.0:

            feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, 1.0))

        else:

            feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, radius))
                
    pharmacophore_writer.write(feats, arquivo_saida)
