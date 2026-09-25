from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

out = Path(r'E:\automated-3d-model-splitting\example\yoshisittingonledge_updated_recovery\boundary_sampled\connector_backing_failure_schematic.png')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
fig.patch.set_facecolor('#f7f8fa')
ax.set_facecolor('#f7f8fa')
ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.7, 5.8); ax.set_aspect('equal'); ax.axis('off')
# Parent material and the cut boundary in section view.
parent = Polygon([(1,1.0),(9.8,1.0),(9.8,4.5),(1,4.5)], closed=True, facecolor='#e8edf4', edgecolor='none')
ax.add_patch(parent)
ax.plot([1,9.8],[4.5,4.5],color='#26384a',lw=3)
ax.text(9.65,4.78,'零件边界截面',ha='right',va='bottom',fontsize=13,color='#26384a')
ax.text(9.5,2.0,'母体材料',ha='right',fontsize=13,color='#596777')
# Show the intended 45-degree backing, but mark that no continuous topology-safe inset ring was produced.
ax.plot([3.1,4.45],[4.5,3.15],color='#1e8e6e',lw=5,solid_capstyle='round')
ax.plot([4.45,7.8],[3.15,3.15],color='#1e8e6e',lw=5,solid_capstyle='round')
ax.text(3.2,3.55,'目标：连续 45° 背衬',rotation=-45,fontsize=12,color='#08785a',ha='center')
ax.annotate('',xy=(4.45,2.8),xytext=(4.45,3.35),arrowprops=dict(arrowstyle='-|>',lw=1.8,color='#2d6cdf'))
ax.text(4.68,3.0,'可用轴向深度约 0.109 mm',fontsize=11,color='#2d6cdf',va='center')
# Failure area: mark only the attempted continuous contour, without asserting its exact local shape.
ax.plot([3.1,4.45],[4.5,3.15],color='#d64545',lw=3,ls=(0,(2,2)),alpha=.95)
ax.plot([4.45,7.8],[3.15,3.15],color='#d64545',lw=3,ls=(0,(2,2)),alpha=.95)
ax.text(6.05,2.55,'未找到可连续打印的拓扑安全内缩环',ha='center',fontsize=12,color='#b42323',weight='bold')
ax.text(5.95,1.83,'这张是截面原理示意；运行未保存失败环的实际三维顶点，\n因此红色虚线不代表模型中的精确轮廓。',ha='center',fontsize=10.5,color='#505b66')
ax.text(0.0,5.5,'本轮拆分失败定位',fontsize=20,weight='bold',color='#1f2937')
ax.text(0.0,5.03,'完整树预检 / 连接器背衬阶段',fontsize=13,color='#4b5563')
ax.text(0.0,0.2,'失败报告：connector boundary has no continuous printable 45-degree backing',fontsize=10.5,color='#b42323')
ax.text(0.0,-0.25,'边界顶点数：510   |   计划背衬深度：0.109 mm   |   输出 3MF：未生成',fontsize=11,color='#374151')
fig.tight_layout(pad=1.1)
fig.savefig(out,bbox_inches='tight')
print(out)
