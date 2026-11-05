"""Prospective real-data task extension; the matrix is fixed before test access."""
import itertools

METHODS=['E','B3','B35','B5','A5']
CASES=[('taxonomy','wordnet_mammal'),('link_prediction','cora_links'),
       ('retrieval','digits_retrieval'),('taxonomy','wordnet_vehicle'),
       ('link_prediction','citeseer_links'),('retrieval','20news_retrieval')]

def generate():
    configs=[]
    for priority,(family,dataset) in enumerate(CASES):
        rates=[.1,1.] if family=='retrieval' else [10.,100.]
        for method in METHODS:
            for slot,(lr,index) in enumerate(itertools.product(rates,range(3))):
                c=dict(suite='new_tasks_v1',family=family,dataset=dataset,
                    model='product_proxy' if family=='retrieval' else 'product_embedding',
                    method=method,epochs=1000,seed=42,dim=8,groups=4,lr=lr,rho=.01,
                    beta=.9,init_K=[.1,1.,3.][index] if method!='E' else 1.,
                    point_decay=0. if method!='E' else [0.,1e-4,1e-2][index],
                    search_slot=slot,search_budget=6,priority=priority,eval_every=25,
                    selection_metric={'taxonomy':'map','link_prediction':'ap','retrieval':'recall1'}[family])
                c['id']=f'new_{family}_{dataset}_{method}_s{slot}'
                configs.append(c)
    return configs
