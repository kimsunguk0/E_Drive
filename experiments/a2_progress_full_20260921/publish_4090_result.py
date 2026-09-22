"""Publish the exact FULL terminal's separately completed RTX4090 verification."""
from pathlib import Path
import json,subprocess

ROOT=Path('/NHNHOME/data/sukim/adcl');REPORT=ROOT/'reports/a2_progress_full_20260921'
def command(args,cwd=ROOT):return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT)
def main():
    result=json.loads((REPORT/'RTX4090_FULL_validation.json').read_text());completion=json.loads((REPORT/'completion.json').read_text())
    assert result['status']=='passed' and result['actual_FULL_terminal_weights']
    assert result['checkpoint_sha256']==completion['checkpoint_sha256']
    assert not command(['git','diff','--cached','--name-only']).strip()
    completion.update(RTX4090_ms=result['largest_clip_median_ms'],RTX4090_p95_ms=result['largest_clip_p95_ms'],RTX4090_full_weights_verified=True)
    (REPORT/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    note='\n추가 검증 완료: chi@192.168.10.102의 RTX4090에서 **동일 FULL terminal**의 전체 B1 BF16 forward '+str(round(result['largest_clip_median_ms'],3))+'ms, p95 '+str(round(result['largest_clip_p95_ms'],3))+'ms. 두 train fixture 각각 warmup30/repeat200, 전처리 제외. [측정·FP32 재현](RTX4090_FULL_validation.json). 앞의 4090 미측정 문구는 패키지 생성 당시 상태이며 이번 기록으로 갱신된다. 공식 업로드는 수행하지 않았다.\n'
    with (REPORT/'PACKAGE_KO.md').open('a') as f:f.write(note)
    paths=[str((REPORT/n).relative_to(ROOT)) for n in ('RTX4090_FULL_validation.json','completion.json','PACKAGE_KO.md')]
    command(['git','add','--',*paths]);command(['git','diff','--cached','--check']);command(['git','commit','-m','Verify exact FULL terminal on RTX4090 and record complete forward latency'])
    work=command(['git','rev-parse','HEAD']).strip();mirror=Path('/home/<B200-USER>/edrive_mirror')
    assert not command(['git','status','--porcelain'],mirror).strip()
    assert command(['git','branch','--show-current'],mirror).strip()=='motiondrive-v2-20260910'
    command(['git','fetch','github','motiondrive-v2-20260910'],mirror);command(['git','merge','--ff-only','FETCH_HEAD'],mirror)
    command(['git','fetch',str(ROOT),work],mirror);command(['git','cherry-pick',work],mirror);command(['git','push','github','HEAD:motiondrive-v2-20260910'],mirror)
    head=command(['git','rev-parse','HEAD'],mirror).strip();remote=command(['git','ls-remote','github','refs/heads/motiondrive-v2-20260910'],mirror).split()[0];assert head==remote
    receipt={'work':work,'mirror':head,'verified':True};(REPORT/'runtime/4090_publication.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
