* Encoding: UTF-8.

DATASET ACTIVATE 数据集4.

UNIANOVA Cluster BY グループ属性① 性別 年齢 職業 日常の自動車利用 府県コード 京都市フラグ 子供フラグ 高齢者フラグ
  /METHOD=SSTYPE(3)
  /INTERCEPT=INCLUDE
  /PRINT HOMOGENEITY
  /CRITERIA=ALPHA(.05)
  /DESIGN=グループ属性① 性別 年齢 職業 日常の自動車利用 府県コード 京都市フラグ 子供フラグ 高齢者フラグ.

* cross tables


CROSSTABS
  /TABLES=Cluster BY グループ属性① 
  /FORMAT=AVALUE TABLES
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.
* グループ属性① stacked chart. 

GGRAPH 
  /GRAPHDATASET NAME="graphdataset" VARIABLES=グループ属性① COUNT()[name="COUNT"] Cluster 
    MISSING=LISTWISE REPORTMISSING=NO 
  /GRAPHSPEC SOURCE=INLINE 
   TEMPLATE=["/Users/gary/Dropbox/tourism survey/SPSS Correlation & "+ 
    "ANOVA/Templates/APA_Styles.sgt"]. 
BEGIN GPL 
  SOURCE: s=userSource(id("graphdataset")) 
  DATA: グループ属性a=col(source(s), name("グループ属性①"), unit.category()) 
  DATA: COUNT=col(source(s), name("COUNT")) 
  DATA: Cluster=col(source(s), name("Cluster"), unit.category()) 
  COORD: rect(dim(1,2), transpose()) 
  GUIDE: axis(dim(1), label("グループ属性①")) 
  GUIDE: axis(dim(2), label("Percentage")) 
  GUIDE: legend(aesthetic(aesthetic.color.interior), label("Cluster")) 
  GUIDE: text.title(label("Cluster precentage group by travel groups")) 
  SCALE: cat(dim(1), include("1", "2", "3", "4", "5")) 
  SCALE: linear(dim(2), include(0)) 
  SCALE: cat(aesthetic(aesthetic.color.interior), include("1", "2", "3")) 
  ELEMENT: interval.stack(position(summary.percent(グループ属性a*COUNT, base.coordinate(dim(1)))), 
    color.interior(Cluster), shape.interior(shape.square)) 
END GPL.


* cross tables

CROSSTABS
  /TABLES=Cluster BY  性別 
  /FORMAT=AVALUE TABLES
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.
* 性別 stacked chart. 
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=性別 COUNT()[name="COUNT"] Cluster MISSING=LISTWISE 
    REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
   TEMPLATE=["/Users/gary/Dropbox/tourism survey/SPSS Correlation & "+
    "ANOVA/Templates/APA_Styles.sgt"].
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: 性別=col(source(s), name("性別"), unit.category())
  DATA: COUNT=col(source(s), name("COUNT"))
  DATA: Cluster=col(source(s), name("Cluster"), unit.category())
  COORD: rect(dim(1,2), transpose())
  GUIDE: axis(dim(1), label("Gender"))
  GUIDE: axis(dim(2), label("Percentage"))
  GUIDE: legend(aesthetic(aesthetic.color.interior), label("Cluster"))
  GUIDE: text.title(label("Cluster precentage grouped by gender"))
  SCALE: cat(dim(1), include("1", "2"))
  SCALE: linear(dim(2), include(0))
  SCALE: cat(aesthetic(aesthetic.color.interior), include("1", "2", "3"))
  ELEMENT: interval.stack(position(summary.percent(性別*COUNT, base.coordinate(dim(1)))), 
    color.interior(Cluster), shape.interior(shape.square))
END GPL.


* cross tables

CROSSTABS
  /TABLES=Cluster BY  年齢 
  /FORMAT=AVALUE TABLES
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.

* 年齢 stacked chart. 
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=年齢 COUNT()[name="COUNT"] Cluster MISSING=LISTWISE 
    REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
   TEMPLATE=["/Users/gary/Dropbox/tourism survey/SPSS Correlation & "+
    "ANOVA/Templates/APA_Styles.sgt"].
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: 年齢=col(source(s), name("年齢"), unit.category())
  DATA: COUNT=col(source(s), name("COUNT"))
  DATA: Cluster=col(source(s), name("Cluster"), unit.category())
  COORD: rect(dim(1,2), transpose())
  GUIDE: axis(dim(1), label("Age levels"))
  GUIDE: axis(dim(2), label("Percentage"))
  GUIDE: legend(aesthetic(aesthetic.color.interior), label("Cluster"))
  GUIDE: text.title(label("Cluster precentage grouped by age"))
  SCALE: cat(dim(1), include("1", "2"))
  SCALE: linear(dim(2), include(0))
  SCALE: cat(aesthetic(aesthetic.color.interior), include("1", "2", "3"))
  ELEMENT: interval.stack(position(summary.percent(年齢*COUNT, base.coordinate(dim(1)))), 
    color.interior(Cluster), shape.interior(shape.square))
END GPL.


* cross tables

CROSSTABS
  /TABLES=Cluster BY 職業 
  /FORMAT=AVALUE TABLES
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.
* 職業 stacked chart. 
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=職業 COUNT()[name="COUNT"] Cluster MISSING=LISTWISE 
    REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
   TEMPLATE=["/Users/gary/Dropbox/tourism survey/SPSS Correlation & "+
    "ANOVA/Templates/APA_Styles.sgt"].
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: 職業=col(source(s), name("職業"), unit.category())
  DATA: COUNT=col(source(s), name("COUNT"))
  DATA: Cluster=col(source(s), name("Cluster"), unit.category())
  COORD: rect(dim(1,2), transpose())
  GUIDE: axis(dim(1), label("職業"))
  GUIDE: axis(dim(2), label("Percentage"))
  GUIDE: legend(aesthetic(aesthetic.color.interior), label("Cluster"))
  GUIDE: text.title(label("Cluster precentage grouped by job type"))
  SCALE: cat(dim(1), include("1", "2"))
  SCALE: linear(dim(2), include(0))
  SCALE: cat(aesthetic(aesthetic.color.interior), include("1", "2", "3"))
  ELEMENT: interval.stack(position(summary.percent(職業*COUNT, base.coordinate(dim(1)))), 
    color.interior(Cluster), shape.interior(shape.square))
END GPL.

* cross tables

CROSSTABS
  /TABLES=Cluster BY 日常の自動車利用
  /FORMAT=AVALUE TABLES
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.

* 日常の自動車利用 stacked chart. 
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=日常の自動車利用 COUNT()[name="COUNT"] Cluster MISSING=LISTWISE 
    REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
   TEMPLATE=["/Users/gary/Dropbox/tourism survey/SPSS Correlation & "+
    "ANOVA/Templates/APA_Styles.sgt"].
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: 日常の自動車利用=col(source(s), name("日常の自動車利用"), unit.category())
  DATA: COUNT=col(source(s), name("COUNT"))
  DATA: Cluster=col(source(s), name("Cluster"), unit.category())
  COORD: rect(dim(1,2), transpose())
  GUIDE: axis(dim(1), label("daily car usage"))
  GUIDE: axis(dim(2), label("Percentage"))
  GUIDE: legend(aesthetic(aesthetic.color.interior), label("Cluster"))
  GUIDE: text.title(label("Cluster precentage grouped by car usage"))
  SCALE: cat(dim(1), include("1", "2"))
  SCALE: linear(dim(2), include(0))
  SCALE: cat(aesthetic(aesthetic.color.interior), include("1", "2", "3"))
  ELEMENT: interval.stack(position(summary.percent(日常の自動車利用*COUNT, base.coordinate(dim(1)))), 
    color.interior(Cluster), shape.interior(shape.square))
END GPL.

* cross tables

CROSSTABS
  /TABLES=Cluster BY 府県コード
  /FORMAT=AVALUE TABLES
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.

* 府県コード stacked chart. 
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=府県コード COUNT()[name="COUNT"] Cluster MISSING=LISTWISE 
    REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
   TEMPLATE=["/Users/gary/Dropbox/tourism survey/SPSS Correlation & "+
    "ANOVA/Templates/APA_Styles.sgt"].
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: 府県コード=col(source(s), name("府県コード"), unit.category())
  DATA: COUNT=col(source(s), name("COUNT"))
  DATA: Cluster=col(source(s), name("Cluster"), unit.category())
  COORD: rect(dim(1,2), transpose())
  GUIDE: axis(dim(1), label("Prefecture code"))
  GUIDE: axis(dim(2), label("Percentage"))
  GUIDE: legend(aesthetic(aesthetic.color.interior), label("Cluster"))
  GUIDE: text.title(label("Cluster precentage grouped by prefecture"))
  SCALE: cat(dim(1), include("1", "2"))
  SCALE: linear(dim(2), include(0))
  SCALE: cat(aesthetic(aesthetic.color.interior), include("1", "2", "3"))
  ELEMENT: interval.stack(position(summary.percent(府県コード*COUNT, base.coordinate(dim(1)))), 
    color.interior(Cluster), shape.interior(shape.square))
END GPL.

* cross tables

CROSSTABS
  /TABLES=Cluster BY 京都市フラグ
  /FORMAT=AVALUE TABLES
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.
* 京都市フラグ stacked chart. 
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=京都市フラグ COUNT()[name="COUNT"] Cluster MISSING=LISTWISE 
    REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
   TEMPLATE=["/Users/gary/Dropbox/tourism survey/SPSS Correlation & "+
    "ANOVA/Templates/APA_Styles.sgt"].
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: 京都市フラグ=col(source(s), name("京都市フラグ"), unit.category())
  DATA: COUNT=col(source(s), name("COUNT"))
  DATA: Cluster=col(source(s), name("Cluster"), unit.category())
  COORD: rect(dim(1,2), transpose())
  GUIDE: axis(dim(1), label("Kyoto or not"))
  GUIDE: axis(dim(2), label("Percentage"))
  GUIDE: legend(aesthetic(aesthetic.color.interior), label("Cluster"))
  GUIDE: text.title(label("Cluster precentage grouped by Kyoto residence"))
  SCALE: cat(dim(1), include("1", "2"))
  SCALE: linear(dim(2), include(0))
  SCALE: cat(aesthetic(aesthetic.color.interior), include("1", "2", "3"))
  ELEMENT: interval.stack(position(summary.percent(京都市フラグ*COUNT, base.coordinate(dim(1)))), 
    color.interior(Cluster), shape.interior(shape.square))
END GPL.


* cross tables

* 4-28
* 修改了kids 和 elderly的 flag，然后重新跑了KMEANS-3 clusters

* cross tables

CROSSTABS
  /TABLES=Cluster BY 子供フラグ
  /FORMAT=AVALUE TABLES
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.
* 子供フラグ stacked chart. 
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=子供フラグ COUNT()[name="COUNT"] Cluster 
    MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
   TEMPLATE=["/Users/gary/Dropbox/tourism survey/SPSS Correlation & "+
    "ANOVA/Templates/APA_Styles.sgt"].
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: 子供フラグ=col(source(s), name("子供フラグ"), unit.category())
  DATA: COUNT=col(source(s), name("COUNT"))
  DATA: Cluster=col(source(s), name("Cluster"), unit.category())
  COORD: rect(dim(1,2), transpose())
  GUIDE: axis(dim(1), label("Kids flag"))
  GUIDE: axis(dim(2), label("Percentage"))
  GUIDE: legend(aesthetic(aesthetic.color.interior), label("Cluster"))
  GUIDE: text.title(label("Cluster precentage grouped by kids flag"))
  SCALE: linear(dim(2), include(0))
  SCALE: cat(aesthetic(aesthetic.color.interior), include("1", "2", "3"))
  ELEMENT: interval.stack(position(summary.percent(子供フラグ*COUNT, base.coordinate(dim(1)))), 
    color.interior(Cluster), shape.interior(shape.square))
END GPL.


* cross tables

CROSSTABS
  /TABLES=Cluster BY 高齢者フラグ
  /FORMAT=AVALUE TABLES
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.
* 高齢者フラグ stacked chart. 
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=高齢者フラグ COUNT()[name="COUNT"] Cluster 
    MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
   TEMPLATE=["/Users/gary/Dropbox/tourism survey/SPSS Correlation & "+
    "ANOVA/Templates/APA_Styles.sgt"].
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: 高齢者フラグ=col(source(s), name("高齢者フラグ"), unit.category())
  DATA: COUNT=col(source(s), name("COUNT"))
  DATA: Cluster=col(source(s), name("Cluster"), unit.category())
  COORD: rect(dim(1,2), transpose())
  GUIDE: axis(dim(1), label("Elderly"))
  GUIDE: axis(dim(2), label("Percentage"))
  GUIDE: legend(aesthetic(aesthetic.color.interior), label("Cluster"))
  GUIDE: text.title(label("Cluster precentage grouped by elderly flag"))
  SCALE: linear(dim(2), include(0))
  SCALE: cat(aesthetic(aesthetic.color.interior), include("1", "2", "3"))
  ELEMENT: interval.stack(position(summary.percent(高齢者フラグ*COUNT, base.coordinate(dim(1)))), 
    color.interior(Cluster), shape.interior(shape.square))
END GPL.
