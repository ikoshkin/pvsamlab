PVObject_=pvModule
  Version=7.4
  Flags=$00908043

  PVObject_Commercial=pvCommercial
    Flags=$0041
    Manufacturer=Solarspace Technology Co., Ltd
    Model=SSA-66HD-625N
    DataSource=RETCT-SST044e-240723
    Width=1.134
    Height=2.382
    Depth=0.030
    Weight=32.500
    NPieces=100
    PriceDate=14/07/24 21:53
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=66
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  FrontSurface=fsNormalGlass
  GRef=1000
  TRef=25.0
  PNom=625.0
  PNomTolLow=-3.00
  PNomTolUp=3.00
  BifacialityFactor=0.792
  Isc=16.040
  Voc=49.58
  Imp=15.190
  Vmp=41.17
  muISC=6.70
  muVocSpec=-128.9
  muPmpReq=-0.312
  RShunt=280
  Rp_0=1070
  Rp_Exp=4.80
  RSerie=0.184
  Gamma=1.063
  muGamma=-0.0006
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=-11998.800
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  CellWidth=0.210
  CellHeight=0.091
  SandiaAMCorr=50.000

  PVObject_IAM=pvIAM
    Flags=$00
    IAMMode=UserProfile
    IAMProfile=TCubicProfile
      NPtsMax=9
      NPtsEff=9
      LastCompile=$B18D
      Mode=3
      Point_1=0.0,1.00000
      Point_2=30.0,0.98200
      Point_3=50.0,0.95800
      Point_4=60.0,0.92600
      Point_5=70.0,0.86100
      Point_6=75.0,0.79400
      Point_7=80.0,0.68500
      Point_8=85.0,0.45500
      Point_9=90.0,0.00000
    End of TCubicProfile
  End of PVObject pvIAM
End of PVObject pvModule
