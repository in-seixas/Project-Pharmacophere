#!/usr/bin/env python
# coding: utf-8

# In[1]:



def libray_pharmit(caminho, tipo, saida):
    
    saida = saida + '.smi'
    
    smiles_actives = []
    smiles_decoys = []
    id_actives = []
    id_decoys = []
    
    if tipo == 'active': 
          
        with open(caminho, 'r') as reader:
            for ids in reader.readlines():
                ids = ids.split(" ")
                actives = str(ids[2]).strip() + '_'+tipo
                id_actives.append(actives)
                smiles = ids[0]
                smiles_actives.append(smiles)      
        
        with open(saida, 'w') as f:
            for index in range(len(id_actives)):
                f.writelines(str(smiles_actives[index]) +'  '+ str(id_actives[index]) + "\n")

    elif tipo == 'decoy':
        
        with open(caminho, 'r') as reader:
            for ids in reader.readlines():
                ids = ids.split()
                decoys = str(ids[1]).strip() + '_'+tipo
                id_decoys.append(decoys)
                decoys = ids[0]
                smiles_decoys.append(decoys)      
        

        with open(saida, 'a') as f:
            for index in range(len(id_decoys)):
                f.writelines(str(smiles_decoys[index]) +'  '+ str(id_decoys[index]) + "\n")


# In[4]:


libray_pharmit('/home/gessualdo/Documentos/maps_fragmap/1e66/decoys_final.ism', 'decoy', '/home/gessualdo/Documentos/maps_fragmap/1e66/library.mol')


# In[2]:



           


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





# In[ ]:




