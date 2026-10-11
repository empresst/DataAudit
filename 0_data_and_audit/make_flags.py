import os
import glob
import polars as pl

B = os.environ.get("DATA_DIR", "./data")
ic_matches = glob.glob(os.environ.get("MIMIC_ICUSTAYS_CSV", "./raw_mimic/icu/icustays.csv*"))
ic = ic_matches[0] if ic_matches else "icustays.csv.gz"
icu=pl.read_csv(ic,columns=['stay_id','intime'],try_parse_dates=True)
cov_in=set(int(x) for x in open('stays_inputevents.txt').read().split())
cov_pe=set(int(x) for x in open('stays_procedureevents.txt').read().split())
v=pl.read_csv('vaso_ext_intervals.csv',has_header=False,new_columns=['stay_id','start','end','itemid'],try_parse_dates=True)
OLD=[221906,221289,221662,221749,222315,227692]
def flags(ids,tag):
    x=v.filter(pl.col('itemid').is_in(ids)).join(icu,on='stay_id')
    end=pl.coalesce(['end','start'])
    lab=(x.filter((pl.col('start')<pl.col('intime')+pl.duration(hours=14))&(end>pl.col('intime')+pl.duration(hours=8))).select('stay_id').unique().with_columns(pl.lit(1).alias(f'label_vaso_{tag}')))
    pre=(x.filter((pl.col('start')<pl.col('intime')+pl.duration(hours=6))&(end>pl.col('intime'))).select('stay_id').unique().with_columns(pl.lit(1).alias(f'already_vaso_{tag}')))
    return lab,pre
l_old,p_old=flags(OLD,'old'); l_ext,p_ext=flags(list(set(OLD+[229617,229630,229631,229632])),'ext')
out=[]
for sp in ['train','val','test']:
    df=pl.read_parquet(f'{B}/{sp}_final_enriched.parquet',columns=['stay_id','anchor_year_group','label_vasopressor','already_vaso','label_intubation','already_vent']).unique('stay_id')
    df=(df.join(l_old,on='stay_id',how='left').join(p_old,on='stay_id',how='left').join(l_ext,on='stay_id',how='left').join(p_ext,on='stay_id',how='left')
        .with_columns([pl.col(c).fill_null(0) for c in ['label_vaso_old','already_vaso_old','label_vaso_ext','already_vaso_ext']])
        .with_columns([pl.col('stay_id').is_in(list(cov_in|cov_pe)).cast(pl.Int8).alias('has_input_data'),pl.lit(sp).alias('split')]))
    out.append(df)
f=pl.concat(out); f.write_parquet('stay_flags.parquet')
print('stays',f.height)
print('VALIDATION old-ID recompute vs parquet (label / already):',
      (f['label_vaso_old']==f['label_vasopressor'].cast(pl.Int64)).mean(), (f['already_vaso_old']==f['already_vaso'].cast(pl.Int64)).mean())
print(f.group_by('anchor_year_group').agg(pl.len().alias('n'),(1-pl.col('has_input_data').mean()).alias('pct_no_data'),
   pl.col('label_vasopressor').mean().alias('vaso_parquet'),pl.col('label_vaso_ext').mean().alias('vaso_ext'),
   pl.col('already_vaso').mean().alias('already_old'),pl.col('already_vaso_ext').mean().alias('already_ext')).sort('anchor_year_group'))
c=f.filter(pl.col('has_input_data')==1)
print('COVERED ONLY'); print(c.group_by('anchor_year_group').agg(pl.len().alias('n'),pl.col('label_vaso_ext').mean().alias('vaso_ext'),pl.col('label_intubation').mean().alias('intub')).sort('anchor_year_group'))
