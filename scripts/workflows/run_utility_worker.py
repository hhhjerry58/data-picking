"""Run one fixed GPU queue for the direct document-utility experiment."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys

from data_picking.utility_data import settings


QUEUES={1:[('general',42),('openwebmath',42),('openstax',43),('finemath',44)],
        2:[('openstax',42),('finemath',43),('general',44),('openwebmath',44)],
        3:[('finemath',42),('general',43),('openwebmath',43),('openstax',44)]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=[1,2,3],required=True)
    p.add_argument('--config',default='configs/document_utility.yaml');p.add_argument('--blob-config',required=True)
    a=p.parse_args();cfg,experiment,_=settings(a.config)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.worker),'Explicitly select the matching free GPU'
    def run(command,name):
        print('START',name,datetime.now(timezone.utc).isoformat(),flush=True)
        with (experiment/'logs'/f'{name}.log').open('a') as log:
            subprocess.run([sys.executable,'scripts/experiment.py',*command,'--config',a.config],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',name,datetime.now(timezone.utc).isoformat(),flush=True)
    if a.worker==1:run(['score-utility','--arm','reference'],'reference-score')
    for arm,seed in QUEUES[a.worker]:
        command=['train-utility','--arm',arm,'--seed',str(seed),'--checkpoint-blob-config',a.blob_config,'--direct-checkpoint-io']
        if (arm,seed)==('general',42):run(command+['--stop-after-steps','32'],f'{arm}-seed{seed}-before-resume-retry')
        run(command,f'{arm}-seed{seed}-train')
        run(['score-utility','--arm',arm,'--seed',str(seed)],f'{arm}-seed{seed}-score')


if __name__=='__main__':main()
