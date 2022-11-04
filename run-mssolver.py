import MSSolver

MSSolver.InitiateMinefield()

top_left, bottom_right = MSSolver.MinesweeperInitiator()

XX,YY,MFgrid = MSSolver.InitiateMinefieldGrid("hard",top_left,bottom_right)

MFgrid = MSSolver.categorizeGrid(XX,YY,MFgrid)

print(MFgrid)