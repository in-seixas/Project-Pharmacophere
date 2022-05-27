from pymol import cmd
from drugpy.ftmap.core import load_atlas
from drugpy.commons import fo_
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from pharmacophore import InteractionKind, Feature, PharmacophoreJsonWriter
import matplotlib.cm as cm
from sklearn import metrics
from scipy import stats
from scipy.spatial import distance
from glob import glob


def get_active_site(caminho_ftmap:str, saida_active_site: str ):

    ftmap = load_atlas(caminho_ftmap, plot = False, table = False)

    objeto = list(ftmap.keys())[0]
    print(objeto)
    
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

    hotspot_max = list(hs_D[0].values() )[0]  #hs mais forte

    for dict_hs in hs_D:
        
        fo = fo_(hotspot_max, dict_hs["Selection"])
        
        if fo >= 0.5: 
                
            join_hs_D.append(dict_hs["Selection"])

        
    all_druggable = " or ".join(join_hs_D)
  

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

    xyz = []
    nivel = []

    for index, _ in enumerate(coordenadas):

        nivel.append(coordenadas[index][0])
        xyz.append(coordenadas[index][2:5])

    return xyz, nivel



def metrics_clusters(X, saida: str, tipo:str, level:int):

    inertias = []
    sils = []
    chs = []
    dbs = []


    size = range(2, 6)

    for k in size:

        k2 = KMeans(random_state= 0, n_clusters= k)
        k2.fit(X)
        inertias.append(k2.inertia_)
        sils.append(

            metrics.silhouette_score(X, k2.labels_)

        )
        chs.append(
            metrics.calinski_harabasz_score(X, k2.labels_)
        )

        dbs.append(metrics.davies_bouldin_score(X, k2.labels_))

    fig, ax = plt.subplots(figsize=(6,4))

    (
        pd.DataFrame(
            {
                "inertia":inertias,
                "silhouete":sils,
                "calinski":chs,
                "davis":dbs,
                "k":size,

            }
        )
        .set_index("k")
        .plot(ax=ax, subplots=True, layout=(2,2))
    )

    fig.savefig(f"{saida}/metrics_to_{tipo}_{level}.png", dpi= 300)    



def points_clusters(points, niveis, k:int, radius_mult):


    kmeans = KMeans(n_clusters = k, random_state=0).fit(points)
    centerofmass = kmeans.cluster_centers_
    centroides = centerofmass
    labels = kmeans.labels_ 
    
    df = pd.DataFrame(points, columns=["X", "Y", "Z"])
    df.loc[:, "Nível"] = niveis
    df.loc[:, "Labels"] = labels
            
    points_and_radius = []
    
    for n in range(0, len(centroides)):
       
        #selecionar os 2 primeiros clusters de maior contorno 
        max_contours = df.groupby(['Labels'])['Nível'].sum()


        df_k = df.loc[df['Labels'] == n]  
     
        distance_ = []

    
        for row in df_k.iterrows():

            
            xyz = list(row[1][0:3])

       
            distance_.append(float(distance.euclidean(xyz, centroides[n])))

        
        points_and_radius.append((centroides[n], statistics.mean(distance_*radius_mult), max_contours[n]))

        points_and_radius.sort(key= lambda x: x[2], reverse= True)

   
    return points_and_radius





def get_centroids_for_pharmacophore(caminho_ftmap: str, 
                            k_acceptor: int, 
                            k_donor: int, 
                            k_apolar:int, 
                            level:int):

    _, hs = get_coord_of_hotspot(caminho_ftmap)

    xyz_acceptor, nivel_acceptor = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "acceptor", "O*+N*", level)
    xyz_donor, nivel_donor = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "donor", "N*+O*", level)
    xyz_apolar, nivel_apolar = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "apolar", "C*", level)

    clusters_acceptor = points_clusters(xyz_acceptor, nivel_acceptor, k_acceptor, 1)
    clusters_donor = points_clusters(xyz_donor, nivel_donor, k_donor, 1)
    clusters_apolar = points_clusters(xyz_apolar, nivel_apolar, k_apolar, 1)

    centers_acceptor = []

    for (x ,y, z), radius, niveis in clusters_acceptor:


        dca = cmd.count_atoms(f' name O*+N* and x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hs}')


        centers_acceptor.append({
            "X, Y, Z":(x,y,z),
            "TIPO":"ACCEPTOR",
            "DC": dca,
            "Raio":radius,
            "Nível":niveis

        })       

    centers_donor = []

    for (x ,y, z), radius, niveis in clusters_donor:

        dcd = cmd.count_atoms(f'name N*+O* and x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hs}')

        centers_donor.append({
            "X, Y, Z":(x,y,z),
            "TIPO":"DONOR",
            "DC": dcd,
            "Raio":radius,
            "Nível":niveis
        })
    

    centers_apolar = []

    for (x ,y, z), radius, niveis in clusters_apolar:

        dch = cmd.count_atoms(f' name C* and x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hs}')

        centers_apolar.append({
            "X, Y, Z":(x,y,z),
            "TIPO":"HYDROFOBIC",
            "DC": dch,
            "Raio":radius,
            "Nível":niveis
        })

    centers_model = list(centers_donor + centers_acceptor + centers_apolar)
    

    #pode ser ordenado pelo nível ou DC...
    centers_model.sort(key= lambda x : x["DC"], reverse= True)


    return centers_model
    
    
    
def build_pharmacohore(caminho_ftmap: str,
                            caminho_saida:str,
                            k_acceptor: int, 
                            k_donor: int, 
                            k_apolar:int, 
                            level:int):
    
    
    centroids = get_centroids_for_pharmacophore(caminho_ftmap, k_acceptor, k_donor, k_apolar, level)
    
    feats = []

    pharmacophore_writer = PharmacophoreJsonWriter()

    for dicts in centroids:
        for key, values in dicts.items():


            if dicts[key] == "ACCEPTOR":
                
                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                radius = list(dicts.values())[3]

                if radius > 1.5:

                    feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, 1.5))

                elif radius < 1.0:

                    feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, 1.0))

                else:

                    feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, radius))



                
            elif dicts[key] == "DONOR":

                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                radius = list(dicts.values())[3]

                
                if radius > 1.5:

                    feats.append(Feature(InteractionKind.DONOR, x, y, z, 1.5))

                elif radius < 1.0:

                    feats.append(Feature(InteractionKind.DONOR, x, y, z, 1.0))

                else:

                    feats.append(Feature(InteractionKind.DONOR, x, y, z, radius))



            elif dicts[key] == "HYDROFOBIC":


                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                radius = list(dicts.values())[3]


                if radius > 1.5:

                    feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, 1.5))

                elif radius < 1.0:

                    feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, 1.0))

                else:

                    feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, radius))



               

                
               
                

    pharmacophore_writer.write(feats, caminho_saida)




def create_pseudoatoms(caminho_ftmap: str, 
                            k_acceptor: int, 
                            k_donor: int, 
                            k_apolar:int, 
                            level:int,
                            saida:str):


    centroids = get_centroids_for_pharmacophore(caminho_ftmap, k_acceptor, k_donor, k_apolar, level)
    
    
    for i, dicts in enumerate(centroids):
        

        if dicts["TIPO"] == "ACCEPTOR":
                            
            x = list(dicts.values())[0][0]
            y = list(dicts.values())[0][1]
            z = list(dicts.values())[0][2]
            radius = list(dicts.values())[3]

            if radius > 1.5:

                cmd.pseudoatom(f"acc_{i}", pos=[x, y, z], elem= "O", chain= "Q", vdw= 1.5)

            elif radius < 1.0:

                cmd.pseudoatom(f"acc_{i}", pos=[x, y, z], elem= "O", chain= "Q", vdw= 1.0)

            else:

                cmd.pseudoatom(f"acc_{i}", pos=[x, y, z], elem= "O", chain= "q", vdw= radius)


            cmd.color("red", f"acc_{i}")

        elif dicts["TIPO"] == "DONOR":

            x = list(dicts.values())[0][0]
            y = list(dicts.values())[0][1]
            z = list(dicts.values())[0][2]
            radius = list(dicts.values())[3]

            if radius > 1.5:

                cmd.pseudoatom(f"don_{i}", pos=[x, y, z], elem= "N", chain= "Q", vdw= 1.5)

            elif radius < 1.0:

                cmd.pseudoatom(f"don_{i}", pos=[x, y, z], elem= "N", chain= "Q", vdw= 1.0)

            else:

                cmd.pseudoatom(f"don_{i}", pos=[x, y, z], elem= "N", chain= "Q", vdw= radius)


            
            cmd.color("blue", f"don_{i}")

        
        elif dicts["TIPO"] == "HYDROFOBIC":

            x = list(dicts.values())[0][0]
            y = list(dicts.values())[0][1]
            z = list(dicts.values())[0][2]
            radius = list(dicts.values())[3]

            if radius > 1.5:

                cmd.pseudoatom(f"apo_{i}", pos=[x, y, z], elem= "C", chain= "Q", vdw= 1.5)

            elif radius < 1.0:

                cmd.pseudoatom(f"apo_{i}", pos=[x, y, z], elem= "C", chain= "Q", vdw= 1.0)

            else:

                cmd.pseudoatom(f"apo_{i}", pos=[x, y, z], elem= "C", chain= "Q", vdw= radius)


            
            cmd.color("yellow", f"apo_{i}")



    cmd.show("spheres", "apo* don* acc*")
    cmd.save(f"{saida}/session2.pse", format= "pse")





#pdbs = ["2am9"]

#for pdb in pdbs:
    #caminhos = [
        #f"/home/gessualdosoj/Documentos/modelos/{pdb}/v2_{pdb}.pdb",
        #f"/home/gessualdosoj/Documentos/modelos/{pdb}/hotspots_active_site.pse",
        #f"/home/gessualdosoj/Documentos/modelos/{pdb}/",
        #f"/home/gessualdosoj/Documentos/modelos/{pdb}/hspharm.json"]
    
    

    #Xa, dc_a = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "acceptor", "O*", 17)
    #Xd, dc_d = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "donor", "N*",  17)
    #Xap, dc_c = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "apolar", "C*", 17)

    
    #metrics_clusters(Xa, caminhos[2], "acceptor", 17)
    #metrics_clusters(Xd, caminhos[2], "donor", 17)
    #metrics_clusters(Xap, caminhos[2], "apolar", 17)


caminhos = [
        "/home/gessualdosoj/Documentos/modelos/2am9/v2_2am9.pdb",
        "/home/gessualdosoj/Documentos/modelos/2am9/hotspots_active_site.pse",
        "/home/gessualdosoj/Documentos/modelos/2am9/",
        "/home/gessualdosoj/Documentos/modelos/2am9/hspharm_17.json"]




#create_pseudoatoms(caminhos[0] , 5, 5, 2, 17, caminhos[2])
build_pharmacohore(caminhos[0], caminhos[3], 2, 3, 2, 17)
create_pseudoatoms(caminhos[0] , 2, 3, 2, 17, caminhos[2])

#get_active_site(caminhos[0], caminhos[2])
