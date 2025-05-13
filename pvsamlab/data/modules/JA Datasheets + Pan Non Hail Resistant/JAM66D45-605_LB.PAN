PVObject_=pvModule
  Version=7.4.0
  Flags=$00900343

  PVObject_Commercial=pvCommercial
    Comment=Shanghai JA Solar Technology Co., Ltd.
    Flags=$0041
    Manufacturer=JA Solar
    Model=JAM66D45-605/LB
    DataSource=TÜV SÜD
    YearBeg=2023
    Width=1.134
    Height=2.382
    Depth=0.030
    Weight=33.100
    NPieces=100
    PriceDate=26/04/20 17:20
    Remarks, Count=6
      Str_1=Cell: 182 x 105 mm
      Str_2=Frame: Aluminum
      Str_3=Structure: 2.0mm double glass module
      Str_4
      Str_5
      Str_6
    End of Remarks
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=66
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  FrontSurface=fsARCoating
  GRef=1000
  TRef=25.0
  PNom=605.0
  PNomTolLow=-3.00
  PNomTolUp=3.00
  BifacialityFactor=0.780
  Isc=16.000
  Voc=47.90
  Imp=15.280
  Vmp=39.60
  muISC=9.60
  muVocSpec=-116.0
  muPmpReq=-0.297
  RShunt=350
  Rp_0=6800
  Rp_Exp=5.50
  RSerie=0.169
  Gamma=1.034
  muGamma=-0.0004
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=11.496
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  SandiaAMCorr=50.000
  RelEffic800=0.24
  RelEffic600=0.17
  RelEffic400=0.04
  RelEffic200=-1.50

  PVObject_IAM=pvIAM
    Flags=$00
    IAMMode=UserProfile
    IAMProfile=TCubicProfile
      NPtsMax=9
      NPtsEff=8
      LastCompile=$B18D
      Mode=3
      Point_1=0.0,1.00000
      Point_2=60.0,1.00000
      Point_3=65.0,0.98800
      Point_4=70.0,0.96500
      Point_5=75.0,0.94200
      Point_6=80.0,0.87400
      Point_7=85.0,0.74400
      Point_8=90.0,0.00000
      Point_9=0.0,0.00000
    End of TCubicProfile
  End of PVObject pvIAM

  OperPoints, list of=12 tOperPoint
    Point_1=False,100,15.0,-2.91,0.00,0.000,0.000,0.00,60.50
    Point_2=False,200,15.0,1.30,0.00,0.000,0.000,0.00,126.24
    Point_3=False,400,15.0,3.33,0.00,0.000,0.000,0.00,257.53
    Point_4=False,600,15.0,3.42,0.00,0.000,0.000,0.00,386.64
    Point_5=False,800,15.0,3.33,0.00,0.000,0.000,0.00,515.07
    Point_6=False,100,25.0,-5.61,0.00,0.000,0.000,0.00,57.17
    Point_7=False,200,25.0,-1.50,0.00,0.000,0.000,0.00,119.32
    Point_8=False,400,25.0,0.04,0.00,0.000,0.000,0.00,242.36
    Point_9=False,600,25.0,0.17,0.00,0.000,0.000,0.00,364.02
    Point_10=False,800,25.0,0.24,0.00,0.000,0.000,0.00,485.70
    Point_11=True,1000,25.0,0.00,47.90,16.000,15.280,39.60,605.09
    Point_12=False,1100,25.0,-0.31,0.00,0.000,0.000,0.00,664.17
  End of List OperPoints
End of PVObject pvModule
