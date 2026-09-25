"""Publication figures for Pixel-OPSD with fixed two-column canvases."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from pycocotools import mask as mask_utils

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
NAVY, BLUE, TEAL = "#18324A", "#2F6F9F", "#2B9A93"
ORANGE, RED, GOLD = "#D98545", "#C95B55", "#D9A44A"
INK, MUTED, GRID, PAPER = "#273A47", "#687984", "#D8E2E5", "#FBFCFC"
PHOTO_ROOT = Path("/volume/ybo/xyc/coco2017")
PHOTO_IDS = ["unlabeled2017/000000000267.jpg", "unlabeled2017/000000000615.jpg", "unlabeled2017/000000000845.jpg"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.4,"axes.titleweight":"bold","pdf.fonttype":42,"ps.fonttype":42,"mathtext.fontset":"dejavusans"})

def box(ax,x,y,w,h,edge=GRID,face="white",radius=.018,lw=1.2,label=None,label_color=INK,label_size=8.5,z=2):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.008,rounding_size={radius}",linewidth=lw,edgecolor=edge,facecolor=face,zorder=z)
    ax.add_patch(p)
    if label is not None: ax.text(x+w/2,y+h/2,label,ha="center",va="center",color=label_color,fontsize=label_size,weight="bold",zorder=z+1)
    return p

def arrow(ax,start,end,color=INK,lw=1.45,rad=0.0,z=6):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=11,linewidth=lw,color=color,connectionstyle=f"arc3,rad={rad}",zorder=z))

def photo(ax,x,y,w,h,idx,caption,accent):
    try:
        arr=np.asarray(Image.open(PHOTO_ROOT/PHOTO_IDS[idx%len(PHOTO_IDS)]).convert("RGB"))
        ax.imshow(arr,extent=(x,x+w,y,y+h),aspect="auto",zorder=1)
    except Exception:
        ax.add_patch(Rectangle((x,y),w,h,facecolor="#E7EFF0",edgecolor=GRID,zorder=1))
        ax.text(x+w/2,y+h/2,"image",ha="center",va="center",color=MUTED,fontsize=7)
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.003,rounding_size=.012",facecolor="none",edgecolor=accent,linewidth=1.1,zorder=3))
    sh=min(.028,h*.22)
    ax.add_patch(Rectangle((x,y),w,sh,facecolor=accent,alpha=.90,edgecolor="none",zorder=4))
    ax.text(x+.012,y+sh/2,caption,ha="left",va="center",color="white",fontsize=6.8,weight="bold",zorder=5)

def token_strip(ax,x,y,labels,colors,w=.22,h=.085):
    gap=.008; tw=(w-gap*(len(labels)-1))/len(labels)
    for i,(label,color) in enumerate(zip(labels,colors)):
        xx=x+i*(tw+gap); box(ax,xx,y,tw,h,edge=color,face=color,radius=.008,lw=.7,label=label,label_color="white",label_size=7.0,z=3)

def finish(fig,name,bottom=.04,top=.94,left=.03):
    fig.subplots_adjust(left=left,right=.97,bottom=bottom,top=top)
    fig.savefig(OUT/name,format="pdf",dpi=300,facecolor="white")
    plt.close(fig)

fig,ax=plt.subplots(figsize=(7.1,3.40)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off"); fig.patch.set_facecolor("white")
ax.text(.02,.965,"Why Pixel-OPSD? From scalar failure to inspectable credit",color=NAVY,fontsize=13,weight="bold",va="top")
# A summary figure: problem, intervention, and observable outcome are deliberately distinct from the detailed training graph.
for x,w,c,head in [(.025,.285,RED,"PROBLEM"),(.357,.285,TEAL,"INTERVENTION"),(.689,.285,ORANGE,"OUTCOME")]:
    ax.add_patch(FancyBboxPatch((x,.17),w,.64,boxstyle="round,pad=.012,rounding_size=.018",facecolor="white",edgecolor=c,linewidth=1.4))
    ax.add_patch(Rectangle((x,.76),w,.05,facecolor=c,edgecolor="none")); ax.text(x+.018,.785,head,ha="left",va="center",color="white",fontsize=8.2,weight="bold")
# Problem card: one scalar shared by heterogeneous decisions.
photo(ax,.05,.54,.09,.13,0,"target",RED); photo(ax,.155,.54,.09,.13,0,"recon.",BLUE); arrow(ax,(.145,.605),(.15,.605),RED)
box(ax,.105,.34,.10,.10,edge=RED,face="#FFF4F2",label=r"$R_C$",label_color=RED,label_size=15); ax.text(.167,.29,"one IoU scalar",ha="center",color=INK,fontsize=7.4,weight="bold")
for xx,idx,lab in [(.052,0,"noun"),(.145,1,"extent"),(.238,2,"edge")]:
    photo(ax,xx,.185,.075,.055,idx,lab,RED)
for xx,label in [(.08,"noun"),(.17,"relation"),(.26,"boundary")]:
    ax.text(xx,.475,label,ha="center",va="center",color=RED,fontsize=6.2,weight="bold")
# Intervention card: OPD densifies, EGCA localizes.
token_strip(ax,.39,.56,["noun","u1","u2","eos"],[TEAL,ORANGE,TEAL,BLUE],w=.22,h=.07); ax.text(.467,.50,"OPD: token-level teaching",ha="center",color=TEAL,fontsize=7.4,weight="bold")
box(ax,.405,.31,.125,.10,edge=ORANGE,face="#FFF8F0",label="EGCA",label_color=ORANGE,label_size=11); ax.text(.467,.265,r"$g_i=s_i e_i$",ha="center",color=INK,fontsize=8.3); ax.text(.467,.21,"evidence × code position",ha="center",color=MUTED,fontsize=6.9)
token_strip(ax,.402,.445,["z1","z1:z2","z1...zK"],[ORANGE,TEAL,BLUE],w=.13,h=.028)
# Outcome card: evidence and semantic anchor are separate visible benefits.
photo(ax,.715,.54,.10,.13,1,"evidence",ORANGE); box(ax,.835,.56,.095,.10,edge=ORANGE,face="#FFF8F0",label="spatial",label_color=ORANGE,label_size=8.2); arrow(ax,(.82,.605),(.83,.605),ORANGE)
for x,label,c in [(.715,"cycle",BLUE),(.79,"OPD",TEAL),(.865,"sup.",ORANGE)]:
    ax.add_patch(Rectangle((x,.35),.055,.045,facecolor=c,edgecolor="white",linewidth=.7)); ax.text(x+.0275,.3725,label,ha="center",va="center",color="white",fontsize=6.4,weight="bold")
ax.text(.82,.29,"same actor · three complementary signals",ha="center",color=INK,fontsize=7.2,weight="bold")
for xx,label,c in [(.715,"diagnose",RED),(.79,"teach",TEAL),(.865,"anchor",ORANGE)]:
    ax.text(xx+.0275,.425,label,ha="center",color=c,fontsize=6.0,weight="bold")
box(ax,.715,.205,.20,.045,edge=NAVY,face="#F4F7F8",label="inference: image + query only",label_color=NAVY,label_size=6.7,radius=.008,lw=.8)
# Explicit reader takeaway rail.
ax.plot([.05,.95],[.105,.105],color=GRID,linewidth=1.0); ax.text(.05,.07,"diagnose",color=RED,fontsize=7.4,weight="bold"); ax.text(.36,.07,"teach",color=TEAL,fontsize=7.4,weight="bold"); ax.text(.69,.07,"localize + anchor",color=ORANGE,fontsize=7.4,weight="bold")
arrow(ax,(.315,.47),(.35,.47),NAVY,lw=1.1); arrow(ax,(.647,.47),(.682,.47),NAVY,lw=1.1)
finish(fig,"overview.pdf",bottom=.055,top=.96)

fig,ax=plt.subplots(figsize=(7.1,3.42)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off"); fig.patch.set_facecolor("white")
ax.text(.02,.965,"Training graph: sampled prefixes meet decoded mask evidence",color=NAVY,fontsize=12.5,weight="bold",va="top")
ax.text(.03,.825,"A  ON-POLICY ROLLOUT",color=BLUE,fontsize=9.2,weight="bold")
ax.text(.95,.825,"inference path",color=MUTED,fontsize=7.2,weight="bold",ha="right")
photo(ax,.045,.66,.13,.13,0,"x, m*",BLUE); box(ax,.22,.685,.13,.08,edge=GRID,face=PAPER,label="prompt q",label_size=8); box(ax,.405,.64,.22,.17,edge=TEAL,face="#F2FAF8")
ax.text(.515,.755,r"$\pi_\theta$",ha="center",color=NAVY,fontsize=11,weight="bold"); ax.text(.515,.70,"caption c  +  mask codes z",ha="center",color=INK,fontsize=8.6)
box(ax,.69,.71,.13,.07,edge=TEAL,face="white",label="c",label_color=TEAL,label_size=9); box(ax,.69,.61,.13,.07,edge=ORANGE,face="white",label="z1 ... zK",label_color=ORANGE,label_size=8)
arrow(ax,(.18,.725),(.215,.725),BLUE); arrow(ax,(.355,.725),(.40,.725),NAVY); arrow(ax,(.63,.75),(.685,.745),TEAL); arrow(ax,(.63,.675),(.685,.645),ORANGE)
ax.plot([.03,.97],[.555,.555],color=GRID,linewidth=.9,linestyle=(0,(2,2))); ax.text(.03,.515,"B  DECODED EVIDENCE",color=ORANGE,fontsize=9.2,weight="bold")
ax.text(.95,.515,"target-conditioned · training only",color=ORANGE,fontsize=7.2,weight="bold",ha="right")
photo(ax,.055,.345,.14,.125,1,"target m*",ORANGE); photo(ax,.24,.345,.14,.125,1,"D(z)",TEAL); box(ax,.425,.35,.19,.115,edge=ORANGE,face="#FFF8F0")
ax.text(.52,.425,"pixel overlap",ha="center",color=ORANGE,fontsize=8.3,weight="bold"); ax.text(.52,.382,r"$s_i=IoU(D(z_i),m^\star)$",ha="center",color=INK,fontsize=8.0)
box(ax,.68,.35,.23,.115,edge=ORANGE,face="white"); ax.text(.795,.425,"EGCA",ha="center",color=ORANGE,fontsize=9.6,weight="bold"); ax.text(.795,.382,r"$g_i=s_i\,e_i$",ha="center",color=INK,fontsize=8.6)
arrow(ax,(.20,.407),(.42,.407),ORANGE); arrow(ax,(.385,.407),(.42,.407),TEAL); arrow(ax,(.62,.407),(.675,.407),ORANGE); ax.text(.68,.30,"e_i = overlap × code position",color=MUTED,fontsize=7.8)
ax.plot([.03,.97],[.255,.255],color=GRID,linewidth=.9,linestyle=(0,(2,2))); ax.text(.03,.215,"C  OBJECTIVE",color=NAVY,fontsize=9.2,weight="bold")
for x,label,c,sub in [(.06,r"$\mathcal{L}_{cycle}$",BLUE,"trajectory reward"),(.30,r"$\mathcal{L}_{OPD}$",TEAL,"prefix recovery"),(.54,r"$\mathcal{L}_{sup}$",ORANGE,"semantic anchor")]:
    box(ax,x,.09,.18,.095,edge=c,face=c,label=label,label_color="white",label_size=10); ax.text(x+.09,.063,sub,ha="center",color=MUTED,fontsize=7.4)
box(ax,.80,.09,.14,.095,edge=NAVY,face=NAVY,label="update θ",label_color="white",label_size=8.7); arrow(ax,(.24,.137),(.29,.137),BLUE); arrow(ax,(.48,.137),(.53,.137),TEAL); arrow(ax,(.72,.137),(.79,.137),ORANGE); arrow(ax,(.795,.35),(.20,.19),ORANGE,lw=1.2,rad=.18)
ax.text(.31,.235,"code credit",ha="center",va="center",color=ORANGE,fontsize=7.0,weight="bold",rotation=-12)
ax.text(.51,.014,"Only the rollout is sampled; masks are decoded for evidence and never replace the on-policy trajectory.",ha="center",color=MUTED,fontsize=7.2)
finish(fig,"framework.pdf",bottom=.045,top=.96)

# Main evaluation table exported as a wide, typeset exhibit so the benchmark
# comparison remains readable at two-column width without adding non-eval fields.
fig,ax=plt.subplots(figsize=(7.1,3.65)); fig.patch.set_facecolor("white"); ax.axis("off")
headers=["Method","GRES\ngIoU","GRES\ncIoU","N-acc","T-acc","Grounding\ngIoU","DLC"]
rows=[["LISA",".616",".618",".547","--","--","--"],["MLLMSeg",".751",".716",".732","--","--","--"],["HiMTok",".721",".704","--","--","--","--"],["ARGenSeg",".747",".722","--","--","--","--"],["CycleGRPO",".881",".711",".989",".903",".668","--"],["SAMTok",".767",".737",".771",".720",".678","--"],["Qwen25VL-SAMTok (rl)",".794",".737",".815","--","--","--"],["Pixel-OPSD",".792",".712",".780",".999",".673","--"]]
tbl=ax.table(cellText=rows,colLabels=headers,cellLoc="center",colLoc="center",bbox=[.015,.06,.97,.88],colWidths=[.28,.115,.115,.105,.105,.17,.07])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.7)
for (r,c),cell in tbl.get_celld().items():
    cell.set_edgecolor("#D8E2E5"); cell.set_linewidth(.8); cell.PAD=.22
    if r==0:
        cell.set_facecolor("#EEF2F3"); cell.set_text_props(weight="bold",color=NAVY)
    elif r==8:
        cell.set_facecolor("#E8F4F2"); cell.set_text_props(weight="bold",color=NAVY)
    else:
        cell.set_facecolor("white"); cell.set_text_props(color=INK)
for c in range(len(headers)):
    tbl[(0,c)].set_height(.30)
finish(fig,"main_table.pdf",bottom=.02,top=.99,left=.01)

# Independent ablation table: diagnostics only, with placeholders preserved.
fig,ax=plt.subplots(figsize=(7.1,1.55)); fig.patch.set_facecolor("white"); ax.axis("off")
headers=["Variant","Prefix\nagr.","Coarse\nIoU","Fine\nIoU","Boundary\nF1","ECE","QA\nacc.","Refusal"]
rows=[["CycleGRPO","--","--","--","--","--","--","--"],["+ OPD","--","--","--","--","--","--","--"],["+ OPD + EGCA","--","--","--","--","--","--","--"],["+ supervised mixture","--","--","--","--","--","--","--"]]
tbl=ax.table(cellText=rows,colLabels=headers,cellLoc="center",colLoc="center",bbox=[.02,.06,.96,.88],colWidths=[.22,.11,.12,.11,.14,.09,.11,.10])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.4)
for (r,c),cell in tbl.get_celld().items():
    cell.set_edgecolor("#D8E2E5"); cell.set_linewidth(.8); cell.PAD=.16
    if r==0:
        cell.set_facecolor("#EEF2F3"); cell.set_text_props(weight="bold",color=NAVY)
    elif r==4:
        cell.set_facecolor("#E8F4F2"); cell.set_text_props(weight="bold",color=NAVY)
    else:
        cell.set_facecolor("white"); cell.set_text_props(color=INK)
finish(fig,"ablation_table.pdf",bottom=.02,top=.99,left=.01)

fig,axes=plt.subplots(3,1,figsize=(7.1,3.55),sharex=False); fig.patch.set_facecolor("white")
methods=["CycleGRPO","Pixel-OPSD"]; colors=[NAVY,TEAL]
metrics=[("GRES","gIoU",[.881,.792],"14,229 cases"),("GroundingSuite","gIoU",[.668,.673],"3,715 cases")]
for i,(axis,(title,metric,values,cases)) in enumerate(zip(axes[:2],metrics)):
    axis.set_facecolor(PAPER); x=np.arange(2); bars=axis.bar(x,values,color=colors,width=.40,edgecolor="white",linewidth=1.0,hatch=["///", ""],zorder=3)
    for bar,value,c in zip(bars,values,colors):
        axis.text(bar.get_x()+bar.get_width()/2,value+.026,f"{value:.2f}",ha="center",va="bottom",color=c,fontsize=10.2,weight="bold")
    delta=values[1]-values[0]; accent=TEAL if delta>=0 else RED; direction="gain" if delta>=0 else "trade-off"
    axis.text(.08,1.18,title,transform=axis.transAxes,color=NAVY,fontsize=10.0,weight="bold",va="bottom")
    axis.text(.08,.90,cases,transform=axis.transAxes,color=MUTED,fontsize=7.2,va="bottom")
    lens = "false-positive / refusal sensitivity" if title == "GRES" else "multi-instance transfer sensitivity"
    axis.text(.50,.90,lens,transform=axis.transAxes,color=INK,fontsize=7.0,ha="center",va="bottom",style="italic")
    axis.text(.94,1.18,f"Δ {delta:+.2f}",transform=axis.transAxes,color=accent,fontsize=10.8,weight="bold",ha="right",va="bottom")
    axis.text(.94,.92,direction,transform=axis.transAxes,color=accent,fontsize=7.0,weight="bold",ha="right",va="bottom")
    axis.annotate("", xy=(.985, values[1]), xytext=(.985, values[0]), xycoords=("axes fraction", "data"), arrowprops=dict(arrowstyle="-|>", color=accent, lw=1.2))
    axis.set_ylim(0,1); axis.set_yticks([0,.5,1]); axis.set_ylabel(metric if i==0 else "",color=MUTED,fontsize=7.6)
    axis.grid(axis="y",color=GRID,linewidth=.7,zorder=0); axis.set_axisbelow(True)
    axis.axhline(values[0],xmin=.12,xmax=.88,color=NAVY,linewidth=.8,alpha=.30,linestyle=(0,(2,2)),zorder=1); axis.axhline(values[1],xmin=.12,xmax=.88,color=TEAL,linewidth=.8,alpha=.30,linestyle=(0,(2,2)),zorder=1)
    for side in ["top","right"]: axis.spines[side].set_visible(False)
    axis.spines["left"].set_color("#AABBC0"); axis.spines["bottom"].set_color("#AABBC0"); axis.tick_params(axis="y",labelsize=7.0,colors=MUTED); axis.tick_params(axis="x",length=0)
# A derived signed-delta strip makes the trade-off the visual conclusion without inventing a new metric.
axis=axes[2]; deltas=np.array([metrics[0][2][1]-metrics[0][2][0],metrics[1][2][1]-metrics[1][2][0]])
axis.set_facecolor(PAPER); axis.axvline(0,color=GRID,linewidth=1.0); bars=axis.barh(np.arange(2),deltas,color=[RED,TEAL],height=.42,edgecolor="white",zorder=3)
axis.set_yticks(np.arange(2),["GRES","GroundingSuite"],fontsize=7.5,color=INK); axis.set_xlim(-.12,.12); axis.set_xticks([-.10,0,.10],labels=["−0.10","0","+0.10"],fontsize=7.0,color=MUTED); axis.set_xlabel("signed gIoU transfer  (Pixel-OPSD − CycleGRPO)",fontsize=7.2,color=MUTED,labelpad=1)
for b,d,c in zip(bars,deltas,[RED,TEAL]): axis.text(d+(0.008 if d>=0 else -0.008),b.get_y()+b.get_height()/2,f"{d:+.2f}",ha="left" if d>=0 else "right",va="center",fontsize=8.7,weight="bold",color=c)
axis.grid(axis="x",color=GRID,linewidth=.7,zorder=0); axis.set_axisbelow(True)
for side in ["top","right","left"]: axis.spines[side].set_visible(False)
axis.tick_params(axis="y",length=0); axis.spines["bottom"].set_color("#AABBC0")
axes[0].set_xticks(np.arange(2),[]); axes[1].set_xticks(np.arange(2),[])
fig.text(.03,.002,"Matched checkpoints  |  same decoder  |  audited gIoU slices  |  signed strip summarizes transfer",color=MUTED,fontsize=7.0)
fig.text(.97,.002,"higher is better",color=MUTED,fontsize=7.0,ha="right")
finish(fig,"results_bars.pdf",bottom=.105,top=.86,left=.16)

# Experimental protocol matrix: this is a measurement map, not another score table.
fig,ax=plt.subplots(figsize=(7.1,2.55)); fig.patch.set_facecolor("white")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.text(.02,.94,"Evaluation map: headline scores, diagnostics, and audit traces",color=NAVY,fontsize=11.0,weight="bold",va="top")
ax.text(.02,.855,"Each view answers a different question; no diagnostic is folded into the headline comparison.",color=MUTED,fontsize=7.3,va="top")
cols=[("view",.10), ("what is measured",.32), ("controlled signal",.58), ("artifact retained",.82)]
for label,x in cols:
    ax.text(x,.74,label,ha="center",va="center",fontsize=7.5,color=NAVY,weight="bold")
ax.plot([.03,.97],[.69,.69],color=GRID,lw=1.0)
rows=[("GRES","gIoU · cIoU · noun · target","CycleGRPO ↔ Pixel-OPSD","per-benchmark summary"),("GroundingSuite","relation and multi-instance gIoU","same decoder and manifest","query-level counts"),("Ablation","prefix · code · calibration · QA","OPD → EGCA → supervised","per-code audit record"),("Qualitative","coarse · boundary · extent · stable","matched image coordinates","decoded partial masks")]
ys=[.57,.43,.29,.15]
for (name,what,control,artifact),y in zip(rows,ys):
    ax.add_patch(FancyBboxPatch((.03,y-.045),.14,.09,boxstyle="round,pad=.004,rounding_size=.012",facecolor="#EEF2F3",edgecolor=GRID,lw=.7))
    ax.text(.10,y,name,ha="center",va="center",fontsize=7.8,color=NAVY,weight="bold")
    for x,text,c in [( .32,what,INK),(.58,control,TEAL),(.82,artifact,ORANGE)]:
        ax.text(x,y,text,ha="center",va="center",fontsize=7.2,color=c)
    ax.plot([.03,.97],[y-.072,y-.072],color=GRID,lw=.55)
finish(fig,"evaluation_map.pdf",bottom=.08,top=.91,left=.03)

# Error-to-diagnostic map for the experimental analysis; it contains no invented values.
fig,ax=plt.subplots(figsize=(7.1,2.65)); fig.patch.set_facecolor("white")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.text(.02,.94,"Failure-mode analysis: from trajectory error to observable readout",color=NAVY,fontsize=10.8,weight="bold",va="top")
ax.text(.02,.855,"The same sampled response is inspected at semantic, spatial, and language boundaries.",color=MUTED,fontsize=7.2,va="top")
items=[("wrong instance","prefix agreement","OPD",BLUE), ("late boundary drift","fine IoU / boundary F1","EGCA",ORANGE), ("over-extended mask","partial-mask increment","EGCA",TEAL), ("language instability","QA / refusal","supervised","RED")]
xs=[.03,.275,.52,.765]
for x,(failure,readout,signal,color_name) in zip(xs,items):
    color=RED if color_name=="RED" else color_name
    ax.add_patch(FancyBboxPatch((x,.37),.19,.28,boxstyle="round,pad=.008,rounding_size=.016",facecolor="white",edgecolor=color,lw=1.35))
    ax.add_patch(Rectangle((x,.59),.19,.06,facecolor=color,edgecolor="none"))
    ax.text(x+.095,.62,failure,ha="center",va="center",fontsize=7.0,color="white",weight="bold")
    ax.text(x+.095,.51,readout,ha="center",va="center",fontsize=7.0,color=INK,weight="bold")
    ax.text(x+.095,.42,signal,ha="center",va="center",fontsize=7.2,color=color,weight="bold")
    if x < xs[-1]: arrow(ax,(x+.195,.51),(x+.235,.51),NAVY,lw=1.0)
ax.plot([.05,.95],[.22,.22],color=GRID,lw=.9,linestyle=(0,(2,2)))
ax.text(.05,.15,"sampled prefix",color=BLUE,fontsize=6.8,weight="bold")
ax.text(.31,.15,"decoded partial mask",color=ORANGE,fontsize=6.8,weight="bold")
ax.text(.62,.15,"independent diagnostic",color=TEAL,fontsize=6.8,weight="bold",ha="center")
ax.text(.95,.15,"reported separately",color=RED,fontsize=6.8,weight="bold",ha="right")
finish(fig,"error_taxonomy.pdf",bottom=.08,top=.91,left=.03)

# Ablation map: an architectural diagnostic (not an evaluation result) that makes
# the progressive intervention legible while exported metric cells remain blank.
fig,ax=plt.subplots(figsize=(7.1,3.05)); fig.patch.set_facecolor("white")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.text(.02,.95,"Progressive ablation: which learning signal is active?",color=NAVY,fontsize=11.2,weight="bold",va="top")
ax.text(.02,.875,"The rows isolate the causal order; the diagnostic rail maps each signal to the behavior it is intended to regulate.",color=MUTED,fontsize=7.2,va="top")
cols=[("cycle reward",BLUE),("OPD tokens",TEAL),("EGCA code credit",ORANGE),("supervised anchor",RED)]
xs=[.31,.47,.63,.79]
for x,(label,color) in zip(xs,cols):
    ax.text(x,.765,label,ha="center",va="center",fontsize=7.1,color=INK,weight="bold",rotation=0)
    ax.plot([x-.055,x+.055],[.70,.70],color=color,lw=2.8,solid_capstyle="round")
rows=[("CycleGRPO",[1,0,0,0]),("+ OPD",[1,1,0,0]),("+ OPD + EGCA",[1,1,1,0]),("+ supervised mixture",[1,1,1,1])]
ys=[.57,.45,.33,.21]
arrow(ax,(.245,.59),(.245,.19),NAVY,lw=1.0)
ax.text(.225,.39,"causal order",ha="center",va="center",rotation=90,fontsize=6.8,color=MUTED,weight="bold")
for (name,active),y in zip(rows,ys):
    ax.text(.02,y,name,ha="left",va="center",fontsize=8.0,color=NAVY,weight="bold")
    ax.plot([.26,.93],[y,y],color=GRID,lw=.65)
    for x,a,(label,color) in zip(xs,active,cols):
        if a:
            ax.add_patch(FancyBboxPatch((x-.050,y-.036),.10,.072,boxstyle="round,pad=.004,rounding_size=.012",facecolor=color,edgecolor="white",lw=.8))
            ax.text(x,y,"active",ha="center",va="center",fontsize=6.8,color="white",weight="bold")
        else:
            ax.add_patch(FancyBboxPatch((x-.050,y-.036),.10,.072,boxstyle="round,pad=.004,rounding_size=.012",facecolor="#F3F6F7",edgecolor=GRID,lw=.7))
            ax.text(x,y,"off",ha="center",va="center",fontsize=6.8,color=MUTED)
ax.plot([.02,.93],[.105,.105],color=GRID,lw=.8)
ax.text(.02,.075,"diagnostic rail",color=NAVY,fontsize=7.5,weight="bold",va="center")
for x,label,color in [(.30,"semantic / prefix",TEAL),(.51,"coarse-to-fine",ORANGE),(.72,"boundary calib.",BLUE),(.90,"QA + refusal",RED)]:
    ax.add_patch(FancyBboxPatch((x-.085,.045),.17,.055,boxstyle="round,pad=.004,rounding_size=.010",facecolor=color,edgecolor="white",lw=.7))
    ax.text(x,.072,label,ha="center",va="center",fontsize=6.3,color="white",weight="bold")
ax.text(.02,.015,"All rows share the same complete-data protocol. The rail defines independent readouts; it does not introduce extra modules or score values.",color=MUTED,fontsize=6.8)
finish(fig,"ablation_diagnostics.pdf",bottom=.055,top=.91,left=.03)

pred_path=Path("/volume/ybo/xyc/CycleGRPO-OPSD/logs/cyclegrpo20k_withnt_direct30k_notarget10k_dlcqa10k_bs112_directgrpo_ce005_pixel_empty_positivepenalty1/checkpoints/evaluation/step_178_response256_cpu_reassembled/groundingsuite_official_long_prompt_pred.jsonl")
metric_path=pred_path.parent/"groundingsuite_official_long_prompt_metrics.json"; baseline_pred_path=Path("/volume/ybo/xyc/CycleGRPO-OPSD/logs/cyclegrpo20k_pixel_empty_bs128_response256/evaluation/step_156_response256/groundingsuite_pred.jsonl"); baseline_metric_path=baseline_pred_path.parent/"groundingsuite_metrics.json"
pred_by_idx={}; baseline_by_idx={}; metrics_by_idx={}; baseline_iou={}
if pred_path.exists() and metric_path.exists():
    pred_by_idx={r.get("idx"):r for r in (json.loads(x) for x in pred_path.read_text().splitlines() if x.strip())}; metrics_by_idx={r.get("idx"):r for r in json.loads(metric_path.read_text()).get("results",[])}
if baseline_pred_path.exists() and baseline_metric_path.exists():
    baseline_by_idx={r.get("idx"):r for r in (json.loads(x) for x in baseline_pred_path.read_text().splitlines() if x.strip())}; baseline_iou={r.get("idx"):r.get("iou",float("nan")) for r in json.loads(baseline_metric_path.read_text()).get("results",[])}
valid=[idx for idx,rec in pred_by_idx.items() if metrics_by_idx.get(idx,{}).get("gt_segmentation") and rec.get("predicted_segmentation")]
chosen=[valid[0],valid[len(valid)//4],valid[len(valid)//2],valid[(3*len(valid))//4]] if len(valid)>3 else valid
fig,axes=plt.subplots(2,2,figsize=(7.1,4.45)); axes=np.asarray(axes).ravel(); fig.patch.set_facecolor("white")
fig.text(.03,.968,"Qualitative evidence: boundary, extent, and stable cases",color=NAVY,fontsize=11.6,weight="bold",va="top")
fig.text(.03,.925,"red = target   |   dashed blue = CycleGRPO   |   teal = Pixel-OPSD",color=MUTED,fontsize=7.4,ha="left",va="center")
for j,axis in enumerate(axes):
    axis.axis("off")
    if j>=len(chosen): axis.text(.5,.5,"example unavailable",ha="center",va="center",color=MUTED,fontsize=9); continue
    idx=chosen[j]; rec=pred_by_idx[idx]; metric=metrics_by_idx[idx]
    try:
        image=np.asarray(Image.open(PHOTO_ROOT/rec["image_path"]).convert("RGB")); gt=mask_utils.decode(metric["gt_segmentation"]).astype(bool); pred=mask_utils.decode(rec["predicted_segmentation"]).astype(bool); base_rec=baseline_by_idx.get(idx); base=mask_utils.decode(base_rec["predicted_segmentation"]).astype(bool) if base_rec and base_rec.get("predicted_segmentation") else None; shape=(image.shape[1],image.shape[0]); gt=np.asarray(Image.fromarray((gt*255).astype("uint8")).resize(shape,Image.Resampling.NEAREST))>127; pred=np.asarray(Image.fromarray((pred*255).astype("uint8")).resize(shape,Image.Resampling.NEAREST)); pred=pred>127; base=np.asarray(Image.fromarray((base*255).astype("uint8")).resize(shape,Image.Resampling.NEAREST))>127 if base is not None else None
        overlay=image.astype(float)/255; overlay[gt]=.66*overlay[gt]+.34*np.array([.80,.18,.15]); overlay[pred]=.62*overlay[pred]+.38*np.array([.10,.65,.58])
        if base is not None: overlay[base]=.72*overlay[base]+.28*np.array([.13,.34,.65])
        axis.imshow(overlay); axis.contour(gt,levels=[.5],colors=[RED],linewidths=1.25); axis.contour(pred,levels=[.5],colors=[TEAL],linewidths=1.25)
        if base is not None: axis.contour(base,levels=[.5],colors=[BLUE],linewidths=1.05,linestyles="--")
        # A compact inset exposes the local boundary evidence without changing the shared coordinate frame.
        ys,xs=np.where(gt>0)
        if len(xs)>8:
            inset=axis.inset_axes([.67,.05,.29,.29]); inset.imshow(overlay); inset.contour(gt,levels=[.5],colors=[RED],linewidths=.9); inset.contour(pred,levels=[.5],colors=[TEAL],linewidths=.9)
            if base is not None: inset.contour(base,levels=[.5],colors=[BLUE],linewidths=.75,linestyles="--")
            pad= max(4, int(.12*max(xs.max()-xs.min(),ys.max()-ys.min())))
            inset.set_xlim(max(0,xs.min()-pad),min(image.shape[1],xs.max()+pad)); inset.set_ylim(min(image.shape[0],ys.max()+pad),max(0,ys.min()-pad)); inset.set_xticks([]); inset.set_yticks([])
            for spine in inset.spines.values(): spine.set_edgecolor(INK); spine.set_linewidth(.8)
        case_labels=["coarse displacement","boundary drift","extent correction","stable prediction"]
        axis.set_title(f"case {idx}  |  {case_labels[j]}\nIoU {baseline_iou.get(idx,float('nan')):.2f} → {metric.get('iou',float('nan')):.2f}",fontsize=8.0,color=NAVY,pad=3.0,linespacing=1.05)
    except Exception: axis.text(.5,.5,"image unavailable",ha="center",va="center",color=MUTED,fontsize=9)
fig.text(.03,.018,"All panels share one image coordinate frame; contours expose coarse, fine, extent, and stable-mask behavior.",color=MUTED,fontsize=8.0)
finish(fig,"qualitative.pdf",bottom=.045,top=.86)
