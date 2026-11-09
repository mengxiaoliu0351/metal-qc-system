"""One bounded current-version system assessment on fixed public historical images."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.screw_entry_regression import ROOT, LIVE, run, summarize
from products.families import read,write
from review.workflow import file_hash

OUTPUT=LIVE/'experiments/final_system_eval_20261108'


def prepare(output):
    if (output/'protocol.json').exists():raise ValueError('Final assessment protocol is already frozen')
    old=read(LIVE/'experiments/screw_entry_regression_v1/protocol.json')
    original=old['samples'];selected=[]
    def take(cohort,count,label=None,predicate=lambda s:True):
        pool=sorted((s for s in original if s['cohort']==cohort and (label is None or s['truth']==label) and predicate(s)),key=lambda s:s['pixel_hash'])
        if len(pool)<count:raise ValueError('Insufficient frozen samples')
        selected.extend(pool[:count])
    take('mvtec_screw',40)
    for label in ('normal','defect'):take('m2_head',20,label)
    for label in ('normal','defect'):take('m2_body',2,label)
    for label in ('normal','defect'):take('csem_uncovered',2,label,lambda s:s['path'].endswith('_501.png'))
    take('appearance_no_quality_truth',4)
    if len(selected)!=92 or len({s['pixel_hash'] for s in selected})!=92:raise ValueError('Sample count/identity mismatch')
    assets=set(old['assets'])
    for product in read(LIVE/'perception/banks/products.json'):
        assets.add('perception/banks/'+product['bank'])
        if product.get('perception_artifact'):
            name='perception/optimized/'+product['perception_artifact'];assets.add(name)
            assets.add('perception/optimized/'+read(LIVE/name)['arrays'])
        if product.get('review_benchmark'):
            assets.update(path.relative_to(LIVE).as_posix() for path in (LIVE/product['review_benchmark']).iterdir() if path.suffix in ('.json','.npz'))
    assets.update(path.relative_to(LIVE).as_posix() for path in (LIVE/'perception').glob('classifier_*.joblib'))
    if (LIVE/'cognition/standard_library.csv').exists():assets.add('cognition/standard_library.csv')
    sources=set(old['sources'])|{'src/review/workflow.py','src/experiments/final_system_eval.py'}
    protocol={**old,'version':'final-system-eval-20261108-v1','created_utc':datetime.now(timezone.utc).isoformat(),
              'samples':selected,'cohort_counts':dict(Counter(s['cohort'] for s in selected)),
              'sources':{name:file_hash(ROOT/name) for name in sorted(sources)},
              'assets':{name:file_hash(LIVE/name) for name in sorted(assets)},'maximum_vlm_calls':20,
              'selection':'all fixed 40 MVTec screw holdout; first pixel hashes of 20 normal/20 defect HEAD; 2/2 BODY; 2/2 CSEM illumination501; 4 appearance crops; no score or correctness selection',
              'scope':'one current deployed-version backend assessment in isolated local storage; existing public historical images; no new blind/factory evaluation',
              'population_limit':'stratified convenience sample, not production prevalence; report each view separately',
              'counterfactuals':'VLM-every-classified-image and manual-every-image are task-count references only, not measured accuracy/time baselines'}
    output.mkdir(parents=True,exist_ok=True);write(output/'protocol.json',protocol)
    print({'fixed_images':92,'cohorts':protocol['cohort_counts'],'maximum_vlm_calls':20},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','summarize']);parser.add_argument('--output',type=Path,default=OUTPUT)
    a=parser.parse_args();{'prepare':prepare,'run':run,'summarize':summarize}[a.action](a.output.resolve())


if __name__=='__main__':main()
