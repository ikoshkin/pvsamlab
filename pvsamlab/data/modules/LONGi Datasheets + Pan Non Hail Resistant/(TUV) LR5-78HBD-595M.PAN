PVObject_=pvModule
  Version=7.2.4
  Flags=$00900043

  PVObject_Commercial=pvCommercial
    Comment=LONGi Green Energy Technology Co., Ltd.
    Flags=$0041
    Manufacturer=Longi Solar
    Model=LR5-78HBD-595M
    DataSource=TüV SüD
    YearBeg=2024
    Width=1.134
    Height=2.465
    Depth=0.030
    Weight=35.100
    NPieces=100
    PriceDate=08/01/20 10:31
    Remarks, Count=4
      Str_1=Frame: Aluminum
      Str_2=Structure: Tempered AR glass
      Str_3=Connections: Doubleglass, Jbox IP 68, MC4 or mateable
      Str_4=Bifacial PERC, Half cut.
    End of Remarks=Bifacial PERC, Half cut.
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=78
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  FrontSurface=fsARCoating
  GRef=1000
  TRef=25.0
  PNom=595.0
  PNomTolLow=0.00
  PNomTolUp=3.00
  BifacialityFactor=0.700
  Isc=13.960
  Voc=53.90
  Imp=13.110
  Vmp=45.39
  muISC=6.56
  muVocSpec=-142.8
  muPmpReq=-0.340
  RShunt=600
  Rp_0=22000
  Rp_Exp=5.50
  RSerie=0.204
  Gamma=1.010
  muGamma=-0.0004
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=3.200
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  CellArea=166.5
  SandiaAMCorr=50.000
  RelEffic800=0.25
  RelEffic600=0.22
  RelEffic400=-0.04
  RelEffic200=-2.11

  PVObject_IAM=pvIAM
    Flags=$00
    IAMMode=UserProfile
    IAMProfile=TCubicProfile
      NPtsMax=9
      NPtsEff=9
      LastCompile=$B18D
      Mode=3
      Point_1=0.0,1.00000
      Point_2=40.0,1.00000
      Point_3=50.0,1.00000
      Point_4=60.0,1.00000
      Point_5=70.0,0.95000
      Point_6=75.0,0.89000
      Point_7=80.0,0.82000
      Point_8=85.0,0.67000
      Point_9=90.0,0.00000
    End of TCubicProfile
  End of PVObject pvIAM

  OperPoints, list of=7 tOperPoint
    Point_1=False,100,25.0,-4.83,0.00,0.000,0.000,0.00
    Point_2=False,200,25.0,-2.11,0.00,0.000,0.000,0.00
    Point_3=False,400,25.0,-0.04,0.00,0.000,0.000,0.00
    Point_4=False,600,25.0,0.22,0.00,0.000,0.000,0.00
    Point_5=False,800,25.0,0.25,0.00,0.000,0.000,0.00
    Point_6=False,1000,25.0,0.00,0.00,0.000,0.000,0.00
    Point_7=False,1100,25.0,-0.03,0.00,0.000,0.000,0.00
  End of List OperPoints
End of PVObject pvModule
