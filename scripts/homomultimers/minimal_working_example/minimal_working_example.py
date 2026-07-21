import pooled_ppi.yeast_pools as yp
for pid in ['o13297', 'p00445']:
    m = yp.get_models(filters=[('af3_id1','==',pid),('af3_id2','==',pid),('input_type','==','pair'),('sample','==',0)])
    yp.save_predictions_db([m.iloc[0].db_id1, m.iloc[0].db_id2], f'{pid}_pair.pdb')