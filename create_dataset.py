import pandas as pd

columns = ['Source Name', 'Event Text', 'Target Name', 'Event Date']
df = pd.read_csv('202201-icews-events.csv', usecols=columns)
df = df.sample(frac=0.90)
print(df)

df['start'] = df['Event Date'].str.split('/').apply(lambda x: f"d{'_'.join(list(reversed(x)))}")
df['end'] = df['start']
df = df.drop(columns=['Event Date'])

# create entities
entities = list(set(df['Source Name'].values).union(set(df['Target Name'].values)))
print('[entities]', len(entities), entities[:10], entities[-10:])
entities_ids = [f'E{i+1}' for i in range(len(entities))]
df_entities = pd.DataFrame({'entities_ids': entities_ids, 'entities': entities})
print(df_entities)
df_entities.to_csv('data/icews22_full/entities.dict', sep='\t', index=False, header=False)

dict_to_replace_values = dict(zip(entities, entities_ids))
# print(dict_to_replace_values)
df = df.replace({'Source Name': dict_to_replace_values, 'Target Name': dict_to_replace_values})
print(df.head())

# create relations
relations = list(set(df['Event Text'].values))
print('[relations]', len(relations), relations[:10], relations[-10:])
relations_ids = [f'R{i+1}' for i in range(len(relations))]
df_relations = pd.DataFrame({'relations_ids': relations_ids, 'relations': relations})
print(df_relations)
df_relations.to_csv('data/icews22_full/relations.dict', sep='\t', index=False, header=False)

dict_to_replace_values = dict(zip(relations, relations_ids))
# print(dict_to_replace_values)
df = df.replace({'Event Text': dict_to_replace_values})
print(df.head())

df.to_csv('data/icews22_full/temporal', sep='\t', index=False, header=False)
