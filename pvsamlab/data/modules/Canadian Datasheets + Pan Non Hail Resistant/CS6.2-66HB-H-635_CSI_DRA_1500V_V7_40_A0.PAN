PVObject_=pvModule
  Version=7.4
  Flags=$00908043

  PVObject_Commercial=pvCommercial
    Comment=www.csisolar.com
    Flags=$0041
    Manufacturer=CSI Solar Co., Ltd.
    Model=CS6.2-66HB-H-635 1500V
    DataSource=CSI Internal Data
    YearBeg=2024
    Width=1.134
    Height=2.382
    Depth=0.040
    Weight=33.400
    NPieces=0
    PriceDate=22/07/16 16:29
    Currency=EUR
    Remarks, Count=5
      Str_1=Frame: Anodized aluminium alloy
      Str_2=Front glass: 2.0 mm heat strengthened glass with anti-reflective coating
      Str_3=Back glass:  2.0 mm heat strengthened glass
      Str_4=Cable: 4.0 mm² (IEC), 10 AWG (UL)
      Str_5=Connector: T6 or MC4-EVO2 or MC4-EVO2A
    End of Remarks=Connector: T6 or MC4-EVO2 or MC4-EVO2A
  End of PVObject pvCommercial

  Technol=mtHIT
  NCelS=66
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  GRef=1000
  TRef=25.0
  PNom=635.0
  PNomTolLow=0.00
  PNomTolUp=1.45
  BifacialityFactor=0.850
  Isc=15.950
  Voc=50.30
  Imp=14.880
  Vmp=42.70
  muISC=7.98
  muVocSpec=-116.0
  muPmpReq=-0.240
  RShunt=200
  Rp_0=50000
  Rp_Exp=5.50
  RSerie=0.153
  Gamma=1.080
  muGamma=-0.0001
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=21.240
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  CellWidth=0.108
  CellHeight=0.189
  SandiaAMCorr=50.000
  RelEffic800=0.40
  RelEffic400=0.10
  RelEffic200=-1.20

  PVObject_IAM=pvIAM
    Flags=$00
    IAMMode=UserProfile
    IAMProfile=TCubicProfile
      NPtsMax=9
      NPtsEff=9
      LastCompile=$B18D
      Mode=3
      Point_1=20.0,1.00000
      Point_2=40.0,1.00000
      Point_3=60.0,1.00000
      Point_4=65.0,0.99000
      Point_5=70.0,0.96000
      Point_6=75.0,0.92000
      Point_7=80.0,0.84000
      Point_8=85.0,0.72000
      Point_9=90.0,0.00000
    End of TCubicProfile
  End of PVObject pvIAM

  OperPoints, list of=4 tOperPoint
    Point_1=True,1000,25.0,0.00,50.80,13.540,13.070,41.70,545.02
    Point_2=False,800,25.0,0.40,0.00,0.000,0.000,0.00,443.02
    Point_3=False,400,25.0,0.10,0.00,0.000,0.000,0.00,220.85
    Point_4=False,200,25.0,-1.20,0.00,0.000,0.000,0.00,108.99
  End of List OperPoints
End of PVObject pvModule
