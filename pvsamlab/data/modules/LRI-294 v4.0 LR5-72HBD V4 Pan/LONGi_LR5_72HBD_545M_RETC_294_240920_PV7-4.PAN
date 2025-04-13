PVObject_=pvModule
  Version=7.4
  Flags=$00908043

  PVObject_Commercial=pvCommercial
    Flags=$0041
    Manufacturer=LONGi
    Model=LR5-72HBD-545M
    DataSource=RETCCT-LRI294-240920
    Width=1.134
    Height=2.278
    Depth=0.030
    Weight=31.800
    NPieces=100
    PriceDate=31/01/24 18:38
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=72
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  FrontSurface=fsNormalGlass
  GRef=1000
  TRef=25.0
  PNom=545.0
  PNomTolLow=0.00
  PNomTolUp=0.92
  BifacialityFactor=0.741
  Isc=13.920
  Voc=49.65
  Imp=13.040
  Vmp=41.80
  muISC=4.30
  muVocSpec = -134
  muPmpReq=-0.340
  RShunt=550
  Rp_0=2150
  Rp_Exp=6.00
  RSerie=0.207
  Gamma=0.979
  muGamma=-0.0003
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=-11998.800
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  CellWidth=0.182
  CellHeight=0.092
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
      Point_2=30.0,1.00000
      Point_3=50.0,0.99700
      Point_4=60.0,0.97200
      Point_5=70.0,0.90200
      Point_6=75.0,0.83000
      Point_7=80.0,0.71000
      Point_8=85.0,0.48800
      Point_9=90.0,0.00000
    End of TCubicProfile
  End of PVObject pvIAM
End of PVObject pvModule
