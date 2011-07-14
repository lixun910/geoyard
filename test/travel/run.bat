SET BASE=Cluster25\
SET MinPts=50
SET Eps=0.13
SET START=1
SET END=30


REM python DataProcess.py Data\CA.shp %MinPts% %Eps%
REM python extractClusterReachPlot.py Data\CA.shp %BASE%ReachPlot_%MinPts%.txt %BASE%Cluster_%MinPts%.txt %START% %END%

REM ================================create alpha shapes
REM set path=d:\programfiles\python24
REM python AlphaShape.py %BASE%

python.exe PlotTravel.py Cluster25\Cluster_25.19.alpha.shp

