import pandas as pd

ddir = '/home/ngaude/workspace/data/cdiscount/'

color_list = set('roug bleu vert jaun noir gris violet blanc orang marron ros beig'.split())

def get_txt_without_color(r):
    txt = (r.Marque+' '+r.Libelle+' '+r.Description).split()
    txt = ['adidi' if w in color_list else w for w in txt]
    return ' '.join(txt)

##############################
# experimental reinforcement #
##############################

# take your best result in a 2nd pass of index searching 
# to "swipe out" aberrant result, based on clustering predicate 
# for item to be guessed... 
# e.g a specific vendor will not propose a product on its own 
# but rather will copy paste description of the same product with
# different colors ... and so on...


#######################
# textify the test now
# for SELF searching purpose
#######################

df = pd.read_csv(ddir+'test_normed.csv',sep=';',names=['Identifiant_Produit','Description','Libelle','Marque','prix']).fillna('')
df['t_index'] = df.apply(get_txt_without_color,axis=1)
df = df[map(lambda t:len(t)>50,df.t_index)]

resultat = pd.read_csv('/home/ngaude/workspace/data/resultat/resultat.best.68053041.csv',sep=';')

df = df.merge(resultat,'left',None,'Identifiant_Produit','Id_Produit')
df['uid'] = df.Id_Categorie.astype(str)+'@'+df.Identifiant_Produit.astype(str)
df = df[['t_index','uid']].sort(['t_index','uid']).reset_index(drop=True)
df.to_csv(ddir+'search_txt_self.tsv',sep='\t',index=False,header=False)






