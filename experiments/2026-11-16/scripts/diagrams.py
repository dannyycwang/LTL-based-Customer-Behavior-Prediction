"""Editable native draw.io diagrams; no flattened raster elements."""
from pathlib import Path
import xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parents[1]/'generated/figures';OUT.mkdir(parents=True,exist_ok=True)
INK='#233344';MUTED='#617487';LINE='#A9B6C2';BLUE='#3A70A5';TEAL='#008B83';AMBER='#BB7C2D';RED='#B66B61'
class Diagram:
    def __init__(self,name,w,h):
        self.doc=ET.Element('mxfile',host='app.diagrams.net',type='device',version='32.6.10')
        dia=ET.SubElement(self.doc,'diagram',id=name,name=name)
        graph=ET.SubElement(dia,'mxGraphModel',dx=str(w),dy=str(h),grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='0',pageScale='1',pageWidth=str(w),pageHeight=str(h),math='0',shadow='0',background='#FFFFFF')
        self.root=ET.SubElement(graph,'root');ET.SubElement(self.root,'mxCell',id='0');ET.SubElement(self.root,'mxCell',id='1',parent='0');self.i=1
        self.name=name
    def cell(self,x,y,w,h,label='',fill='none',stroke='none',size=16,color=INK,bold=False,align='center',extra=''):
        self.i+=1
        style=f'rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1;fontFamily=Helvetica;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};align={align};verticalAlign=middle;spacing=0;{extra}'
        c=ET.SubElement(self.root,'mxCell',id=str(self.i),value=label,style=style,vertex='1',parent='1')
        ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
        return str(self.i)
    def text(self,x,y,w,h,label,size=16,color=INK,bold=False,align='left',extra=''):
        return self.cell(x,y,w,h,label,size=size,color=color,bold=bold,align=align,extra=extra)
    def edge(self,pts,color=LINE,width=1.4,arrow=True,dashed=False):
        self.i+=1
        style=f'endArrow={"block" if arrow else "none"};endFill=1;endSize=6;html=1;rounded=0;strokeColor={color};strokeWidth={width};dashed={int(dashed)};'
        c=ET.SubElement(self.root,'mxCell',id=str(self.i),style=style,edge='1',parent='1')
        g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
        ET.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),attrib={'as':'sourcePoint'})
        ET.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),attrib={'as':'targetPoint'})
        if len(pts)>2:
            a=ET.SubElement(g,'Array',attrib={'as':'points'})
            for x,y in pts[1:-1]:ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
    def write(self):
        ET.indent(self.doc)
        ET.ElementTree(self.doc).write(OUT/f'{self.name}.drawio',encoding='utf-8',xml_declaration=True)

# Fig. 1: a counterfactual pair isolates order while holding prefix and counts fixed.
d=Diagram('fig1_patterns',600,290)
d.text(0,0,380,24,'Same prefix and action counts',14,bold=True)
d.cell(27,40,153,117,fill='#EFF4F9',stroke='none')
d.text(27,25,153,18,'raw prefix  m = 3',13,BLUE,align='center')
d.text(2,64,24,24,'s₁',16,bold=True);d.text(2,116,24,24,'s₂',16,bold=True)
xs=[37,87,137,209,259,309]
colors={'B':('#F1F4F7',MUTED),'D':('#E7EFF8',BLUE),'A':('#DDF1EB',TEAL),'R':('#F7EAE6',RED)}
for row,seq in enumerate(['BDBDAR','BDBADR']):
    y=61+52*row
    for i,(x,a) in enumerate(zip(xs,seq)):
        f,c=colors[a];d.cell(x,y,36,30,a,fill=f,stroke=c,size=16,color=c,bold=True,extra='arcSize=12;rounded=1;')
        if i<5:d.edge([(x+37,y+15),(xs[i+1]-3,y+15)],LINE,1,True)
    d.edge([(348,y+15),(394,y+15)],MUTED,1.2,True)
for j,x in enumerate([411,465,519],1):d.text(x,24,43,22,f'φ<sub>{j}</sub>',17,align='center')
d.text(399,0,174,23,'Pattern vector',14,bold=True,align='center')
for row,vals in enumerate([[1,1,1],[1,0,0]]):
    for j,v in enumerate(vals):d.cell(411+54*j,61+52*row,43,30,str(v),fill='#DDF1EB' if v else '#F6F7F9',stroke=TEAL if v else '#D5DCE2',size=16,color=TEAL if v else MUTED,bold=True)
d.text(27,165,330,24,'Both: B×2 · D×2 · A×1 · R×1',13,MUTED)
d.edge([(0,199),(575,199)],'#D4DDE4',1,False)
d.text(0,207,150,25,'φ₁ = F A',14,align='center',extra='fontFamily=Times New Roman;')
d.text(150,207,220,25,'φ₂ = F(D ∧ X A)',14,align='center',extra='fontFamily=Times New Roman;')
d.text(370,207,205,25,'φ₃ = F(A ∧ X R)',14,align='center',extra='fontFamily=Times New Roman;')
for x,w,label in [(0,140,'B  browse'),(140,140,'D  detail'),(280,150,'A  add cart'),(430,145,'R  remove')]:
    d.text(x,237,w,22,label,13,MUTED,align='center')
# A compact column-width composition keeps final-paper labels at readable size.
for e in d.doc.iter():
    for a,scale in [('x',.72),('y',.85),('width',.72),('height',.85)]:
        if a in e.attrib:e.set(a,str(float(e.get(a))*scale))
d.write()

# Fig. 2: training provenance is separated from the two prediction paths.
d=Diagram('fig2_framework',1200,445)
d.text(4,0,245,28,'(a) Observed clickstream',19,bold=True)
d.text(292,0,470,28,'(b) Two complementary representations',19,bold=True)
d.text(960,0,230,28,'(c) Purchase prediction',19,bold=True)
# input
d.cell(4,55,244,236,fill='#FAFBFC',stroke='#D4DDE4',extra='rounded=1;arcSize=5;')
d.text(20,68,210,28,'Events available at boundary o',17,bold=True,align='center')
for i,a in enumerate(['B','D','B','⋯','A','R']):
    x=20+35*i
    f,c=colors.get(a,('#FAFBFC',MUTED));d.cell(x,122,29,30,a,fill=f,stroke=c if a!='⋯' else 'none',size=17,color=c,bold=True)
d.text(20,100,101,18,'first m events',13,BLUE,align='center')
d.edge([(18,159),(118,159)],BLUE,2,False)
d.text(20,175,210,24,'s<sup>≤o</sup> = (e₁, …, eₒ)',19,align='center',extra='fontFamily=Times New Roman;')
d.edge([(20,210),(232,210)],'#D4DDE4',1,False)
d.text(20,222,210,26,'Full observation   o = T',16,align='center')
d.text(20,252,210,24,'Fixed origin          o = 10',16,align='center')
# upper neural path
d.cell(296,58,178,74,fill='#EFF4FA',stroke='#B8CADB',extra='rounded=1;arcSize=5;')
d.text(302,67,166,27,'Raw prefix',19,BLUE,bold=True,align='center')
d.text(302,99,166,24,'x⁽ᵐ⁾ = first m events',16,BLUE,align='center',extra='fontFamily=Times New Roman;')
d.cell(528,58,188,74,fill='#EFF4FA',stroke='#B8CADB',extra='rounded=1;arcSize=5;')
d.text(535,67,174,27,'Sequence encoder',19,BLUE,bold=True,align='center')
d.text(535,99,174,24,'bounded neural input',16,BLUE,align='center')
d.edge([(248,107),(272,107),(272,95),(296,95)],BLUE,1.8)
d.edge([(474,95),(528,95)],BLUE,1.8)
d.edge([(716,95),(1004,95),(1004,177)],BLUE,1.8)
d.text(768,66,159,24,'raw embedding h',16,BLUE,align='center',extra='fontFamily=Times New Roman;')
# lower symbolic path
d.cell(296,210,178,92,fill='#F0F7F4',stroke='#ABD2C8',extra='rounded=1;arcSize=5;')
d.text(301,219,168,27,'Trace abstraction ρ',18,TEAL,bold=True,align='center')
d.text(303,248,164,42,'event-aligned<br>proposition sets σₜ',16,TEAL,align='center')
d.cell(528,190,222,132,fill='#F0F7F4',stroke='#ABD2C8',extra='rounded=1;arcSize=5;')
d.text(535,198,208,27,'Temporal pattern queries',18,TEAL,bold=True,align='center')
d.text(539,235,200,31,'zₖ = 1[τ<sup>≤o</sup> ⊨ φₖ]',21,TEAL,align='center',extra='fontFamily=Times New Roman;')
d.text(538,279,202,24,'finite-trace satisfaction',15,TEAL,align='center')
d.edge([(248,236),(272,236),(272,256),(296,256)],TEAL,1.8)
d.edge([(474,256),(528,256)],TEAL,1.8)
d.text(482,225,37,20,'τ<sup>≤o</sup>',18,TEAL,align='center',extra='fontFamily=Times New Roman;')
# projection into shared pattern coordinates
d.text(783,190,147,25,'Satisfaction vector',17,TEAL,bold=True,align='center')
for j,label in enumerate(['z₁','z₂','⋯','z<sub>K</sub>']):
    d.cell(783+37*j,238,31,35,label,fill='#DDF1EB',stroke='#ABD2C8',size=18,color=TEAL,extra='fontFamily=Times New Roman;')
d.text(779,286,157,25,'K = 10 coordinates',15,TEAL,align='center')
d.edge([(750,256),(779,256)],TEAL,1.8)
# fusion
d.cell(960,177,89,112,fill='#FFFFFF',stroke=INK,extra='rounded=1;arcSize=5;')
d.text(969,188,71,30,'Fusion',18,bold=True,align='center')
d.text(968,224,73,48,'feature<br>projection<br>+ classifier',14,MUTED,align='center')
d.edge([(932,256),(960,256)],TEAL,1.8)
d.edge([(1049,232),(1084,232)],INK,1.8)
d.cell(1084,197,103,70,fill='#F3F5F7',stroke='#CCD5DE',extra='rounded=1;arcSize=5;')
d.text(1090,206,91,22,'Purchase',17,bold=True,align='center')
d.text(1090,234,91,23,'score ŷ',18,align='center',extra='fontFamily=Times New Roman;')
# formula learning separated below; native dashed route into checker
d.cell(296,355,635,72,fill='#F7F8FA',stroke='#C4CED7',extra='dashed=1;rounded=1;arcSize=5;')
d.text(308,364,171,21,'LEARN ONCE',13,MUTED,bold=True)
d.text(308,389,171,24,'development traces + labels',14,MUTED)
d.cell(502,369,175,43,'Formula learner',fill='#FFFFFF',stroke='#C4CED7',size=17,color=MUTED)
d.cell(722,369,194,43,'Frozen basis Φ, K = 10',fill='#FFFFFF',stroke='#C4CED7',size=17,color=MUTED)
d.edge([(478,390),(502,390)],MUTED,1.3)
d.edge([(677,390),(722,390)],MUTED,1.3)
d.edge([(819,369),(819,337),(640,337),(640,322)],MUTED,1.4,dashed=True)
d.text(4,351,244,72,'m = 10 in every short branch<br><br>The boundary limits both paths.',15,MUTED,align='center')
d.write()

# Validate native editable structure: distinct labels, geometries, and no images.
for p in OUT.glob('*.drawio'):
    r=ET.parse(p).getroot();cells=r.findall('.//mxCell');ids=[c.get('id') for c in cells]
    assert len(ids)==len(set(ids))
    assert not any('image=' in (c.get('style') or '') for c in cells)
    assert all(c.find('mxGeometry') is not None for c in cells if c.get('vertex') or c.get('edge'))
    print(p.name,len(cells),'editable cells')
