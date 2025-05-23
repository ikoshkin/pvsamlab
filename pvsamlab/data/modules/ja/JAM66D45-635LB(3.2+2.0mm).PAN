﻿PVObject_=pvModule
  Version=7.4.8
  Flags=$00900343

  PVObject_Commercial=pvCommercial
    Flags=$0041
    Manufacturer=JA Solar
    Model=JAM66D45-635/LB(3.2+2.0mm)
    DataSource=JA internal
    Width=1.134
    Height=2.382
    Depth=0.030
    Weight=41.000
    NPieces=100
    PriceDate=17/06/21 18:22
    Remarks, Count=1
      Str_1=3.2mm+2.0mm
    End of Remarks=3.2mm+2.0mm
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=66
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  FrontSurface=fsARCoating
  GRef=1000
  TRef=25.0
  PNom=635.0
  PNomTolLow=0.00
  PNomTolUp=3.00
  BifacialityFactor=0.807
  Isc=16.210
  Voc=49.10
  Imp=15.510
  Vmp=40.94
  muISC=7.01
  muVocSpec=-127.5
  muPmpReq=-0.290
  RShunt=600
  Rp_0=150000
  Rp_Exp=5.50
  RSerie=0.153
  Gamma=1.054
  muGamma=-0.0004
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=-11998.800
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  SandiaAMCorr=50.000
  RelEffic800=0.52
  RelEffic600=0.86
  RelEffic400=0.45
  RelEffic200=-1.00

  PVObject_IAM=pvIAM
    Flags=$00
    IAMMode=UserProfile
    IAMProfile=TCubicProfile
      NPtsMax=9
      NPtsEff=9
      LastCompile=$B18D
      Mode=3
      Point_1=0.0,1.00000
      Point_2=50.0,1.00000
      Point_3=60.0,1.00000
      Point_4=65.0,1.00000
      Point_5=70.0,0.97100
      Point_6=75.0,0.93300
      Point_7=80.0,0.86000
      Point_8=85.0,0.76300
      Point_9=90.0,0.00000
    End of TCubicProfile
  End of PVObject pvIAM

  OperPoints, list of=5 tOperPoint
    Point_1=True,1000,25.0,0.00,49.10,16.210,15.510,40.94,634.98
    Point_2=False,800,25.0,0.52,0.00,0.000,0.000,0.00,506.83
    Point_3=False,400,25.0,0.45,0.00,0.000,0.000,0.00,253.24
    Point_4=False,600,25.0,0.86,0.00,0.000,0.000,0.00,381.41
    Point_5=False,200,25.0,-1.00,0.00,0.000,0.000,0.00,124.79
  End of List OperPoints
End of PVObject pvModule
