"""Portable inference closure, extended from the already validated FULL bundle."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/a2_progress_full_20260921'))
import bundle as old

def build(destination,checkpoint):
    dest=Path(destination);record=old.build(dest,Path(checkpoint))
    for name in ('arm_model.py','fine_motion.py','infer_fourarm.py'):
        source=HERE/name;target=dest/'code'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(source.read_text())
        record['files'][str(source.relative_to(ROOT))]={'repository_sha256':old.sha(source),
            'portable_sha256':old.sha(target),'root_path_rewritten_without_graph_changes':False}
    docker=dest/'code/experiments/a2_progress_full_20260921/Dockerfile'
    docker.write_text(docker.read_text().replace('a2_progress_full_20260921/infer_full.py',
                                               'a2_progress_fourarm_20260921/infer_fourarm.py'))
    key=str(docker.relative_to(dest/'code'))
    record['files'][key]['portable_sha256']=old.sha(docker)
    record['files'][key]['entrypoint_updated_to_explicit_fourarm_loader']=True
    readme=(dest/'README.md').read_text().replace('A2-H4-PROGRESS-FULL inference','Selected single-model PROGRESS stage2 inference')
    readme+='\nThe checkpoint carries an explicit execution configuration. A wrong same-key history feature source is rejected by its persistent signature. No second checkpoint or teacher is loaded.\n'
    (dest/'README.md').write_text(readme)
    (dest/'source_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--destination',required=True);ap.add_argument('--checkpoint',required=True)
    args=ap.parse_args();print(json.dumps(build(args.destination,args.checkpoint)))
