import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
d=pd.read_csv("dev_events.csv"); Ws=[50,100,250,500,1000,4000]
def ms(x):
    m=x.mean(); se=x.std(ddof=1)/np.sqrt(len(x)); return m,1.96*se
sty={"rolling":("#0072B2","o","-","Rolling"),"periodic":("#D55E00","s","-","Periodic ($W/4$ rebuild)"),
     "gated1L":("#009E73","^","--","Gated-1L ($K$=1000)"),"gated2L":("#CC79A7","D","--","Gated-2L ($K$=1000)")}
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1),sharey=True)
for a,sh,t in zip(ax,["level1.0","level0.75"],[r"$\delta=1$",r"$\delta=0.75$"]):
    s=d[d["shift"]==sh]
    st=s[s.pol=="static"].coverage.mean(); a.axhline(st,color="0.3",lw=1,ls=":"); 
    a.text(4000,st+0.02,"static",ha="right",color="0.3",fontsize=8)
    for p,(c,mk,ls,lab) in sty.items():
        q=s[(s.pol==p)&((s.K==1000)|(s.K==0))]
        r=[ms(q[q.W==w].coverage) for w in Ws]
        m=np.array([x[0] for x in r]); e=np.array([x[1] for x in r])
        a.errorbar(Ws,m,yerr=e,color=c,marker=mk,ls=ls,ms=4,lw=1.3,capsize=2,label=lab)
    a.set_xscale("log"); a.set_xticks(Ws); a.set_xticklabels(Ws); a.minorticks_off()
    a.set_xlabel("window $W$ (bars)"); a.set_title(t,fontsize=9); a.set_ylim(0,1.05)
ax[0].set_ylabel("alert coverage of first 500 bars"); ax[0].legend(frameon=False,fontsize=7.5,loc="lower right")
plt.tight_layout(); plt.savefig("fig_coverage.pdf"); plt.close()

i=pd.read_csv("if_events.csv"); i[["v","T"]]=i.variant.str.split("|",expand=True); i["T"]=i["T"].astype(int)
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True)
for a,sh,t in zip(ax,["level0.75","level1.0"],[r"$\delta=0.75$",r"$\delta=1$"]):
    s=i[i["shift"]==sh]; x=np.arange(3); w=0.36
    for k,(v,c,lab) in enumerate([("static","#999999","Frozen forest (matched rate)"),("adaptive","#0072B2","Adaptive forest")]):
        r=[ms(s[(s.v==v)&(s["T"]==T)].coverage) for T in [500,1000,1500]]
        a.bar(x+(k-0.5)*w,[z[0] for z in r],w,yerr=[z[1] for z in r],color=c,capsize=2,label=lab,error_kw={"lw":1})
    a.set_xticks(x); a.set_xticklabels([500,1000,1500]); a.set_xlabel("training window $T$ (bars)"); a.set_title(t,fontsize=9)
ax[0].set_ylabel("alert coverage of first 500 bars"); ax[0].legend(frameon=False,fontsize=7.5,loc="upper left")
plt.tight_layout(); plt.savefig("fig_if.pdf"); plt.close()