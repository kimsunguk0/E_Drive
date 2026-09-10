from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
report=json.loads((root/'temporal_analysis_v1/aggregate.json').read_text())
labels=['A: repeated current','B: real history','C: history + common status']
means=np.array([report['arms'][a]['metrics']['d3']['mean'] for a in 'ABC'])
ci=np.array([report['arms'][a]['metrics']['d3']['ci95'] for a in 'ABC'])
oracle=np.array([report['arms'][a]['metrics']['oracle']['mean'] for a in 'ABC'])
x=np.arange(3)
fig,ax=plt.subplots(figsize=(9,5.2),layout='constrained')
ax.bar(x-.17,means,.32,color='#2563eb',label='Selected trajectory D3')
ax.bar(x+.17,oracle,.32,color='#94a3b8',label='Best candidate D3 (GT oracle)')
ax.errorbar(x-.17,means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='none',ecolor='#163d76',capsize=5,lw=1.4)
for at,val in zip(x-.17,means):ax.annotate(f'{val:.3f}',(at,val),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
for at,val in zip(x+.17,oracle):ax.annotate(f'{val:.3f}',(at,val),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
ax.axhline(.15,color='#dc2626',ls='--',lw=1.3,label='Target D3 = 0.15')
ax.set(xticks=x,xticklabels=labels,ylabel='Official weighted D3 (m); lower is better',ylim=(0,1.40),title='Temporal comparison: target not reached')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.15)
ax.set_axisbelow(True)
ax.legend(loc='upper right',frameon=False,fontsize=9)
fig.text(.015,-.035,'Same public initialization, split and 2,000-step budget; terminal B8 / TUNE 1,998 rows.\nError bars: 20,000 session-bootstrap 95% intervals (11 sessions, one training seed). Oracle uses GT and is not an inference score.',fontsize=9,color='#475569')
out=root/'temporal_analysis_v1/temporal_comparison.png'
fig.savefig(out,dpi=170,bbox_inches='tight')
print(out)
