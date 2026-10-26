"""Append screened coordinates and follow-up tests to the experiment ledger."""

import json
from pathlib import Path


def main():
    ledger=Path('results/misogyny_feature_ledger.jsonl')
    existing=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    keys={(r.get('test'),r.get('kind'),r.get('layer'),r.get('neuron'),r.get('feature_id'),
           r.get('scenario'),r.get('gender'),r.get('condition')) for r in existing}
    probe=json.loads(Path('results/misogyny_probe.json').read_text())
    rows=[]
    for kind in ('native','sae'):
        for rank,row in enumerate(probe['selected'][kind],1):
            item={'test':'targeted_activation_screen','kind':kind,'rank':rank,
                  'source_axis':row['source_axis'],'status':'correlation only',
                  'screen_metrics':row}
            if kind=='native':item.update(layer=row['layer'],neuron=row['neuron'])
            else:item['feature_id']=row['feature_id']
            rows.append(item)
    for row in probe['random_native']:
        rows.append({'test':'random_exploration','kind':'native','layer':row['layer'],
                     'neuron':row['neuron']})
    for fid in probe['random_sae']:
        rows.append({'test':'random_exploration','kind':'sae','feature_id':fid})
    for layer,neuron in ((21,174),(20,6729)):
        rows.append({'test':'gender_token_activation_screen','kind':'native',
                     'layer':layer,'neuron':neuron,
                     'status':'token-level correlation; see fresh lexical follow-up'})
    for i,row in enumerate(json.loads(Path('results/misogyny_attention.json').read_text())['generations']):
        if row['condition']!='baseline':
            rows.append({'test':'attention_direction','kind':'attention','layer':row['layer'],
                         'scenario':row['scenario'],'gender':row['gender'],
                         'condition':row['condition'],'vector_norm':row['vector_norm']})
    for row in [json.loads(x) for x in Path('results/misogyny_patch_controls.jsonl').read_text().splitlines()]:
        if row['condition']!='baseline':
            rows.append({'test':'patch_control','kind':'residual','layer':18,
                         'scenario':row['scenario'],'gender':row['recipient_gender'],
                         'condition':row['condition'],'donor_distance':row['donor_distance']})
    added=0
    with ledger.open('a') as out:
        for row in rows:
            key=(row.get('test'),row.get('kind'),row.get('layer'),row.get('neuron'),
                 row.get('feature_id'),row.get('scenario'),row.get('gender'),row.get('condition'))
            if key in keys:continue
            out.write(json.dumps(row,ensure_ascii=False)+'\n');keys.add(key);added+=1
    print('appended',added,'entries to',ledger)


if __name__=='__main__':main()
