PVObject_=pvModule
  Version=7.2.2
  Flags=$00900643

  PVObject_Commercial=pvCommercial
    Comment=Solarspace
    Flags=$0041
    Manufacturer=Solarspace
    Model=SSA-66HD-625N-new
    DataSource=Manufacturer
    YearBeg=2025
    Width=1.134
    Height=2.382
    Depth=0.030
    Weight=32.500
    NPieces=100
    PriceDate=29/09/24 09:05
  End of PVObject pvCommercial

  Technol=mtSiMono
  NCelS=66
  NCelP=2
  NDiode=3
  SubModuleLayout=slTwinHalfCells
  GRef=1000
  TRef=25.0
  PNom=625.0
  PNomTolLow=0.00
  PNomTolUp=1.50
  BifacialityFactor=0.790
  Isc=16.280
  Voc=48.92
  Imp=15.500
  Vmp=40.36
  muISC=7.33
  muVocSpec=-140.0
  muPmpReq=-0.300
  RShunt=300
  Rp_0=1200
  Rp_Exp=5.50
  RSerie=0.176
  Gamma=1.069
  muGamma=-0.0004
  VMaxIEC=1500
  VMaxUL=1500
  Absorb=0.90
  ARev=3.200
  BRev=19.248
  RDiode=0.010
  VRevDiode=-0.70
  AirMassRef=1.500
  CellArea=190.9
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
      Point_2=30.0,0.99600
      Point_3=50.0,0.99300
      Point_4=60.0,0.98500
      Point_5=70.0,0.95600
      Point_6=75.0,0.88900
      Point_7=80.0,0.77800
      Point_8=85.0,0.63500
      Point_9=90.0,0.00000
    End of TCubicProfile
  End of PVObject pvIAM
End of PVObject pvModule
