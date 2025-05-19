real :: start_time, end_time, elapsed_time, mid_time

integer :: vals(8)

integer*4, parameter :: N=10, P=100, rounding=4

character(len=256) :: filename,string_n,string_p

integer*4:: i, j, i1, j1, comm_gen, init_cond, dummy, number_fires

real*8,dimension(N):: C, M, R, L

real*8,dimension(N):: X,dX,Xout, v,dv,vout,dvt,dvm,vt, Xtot

real*8 :: inv_firetime, ave_firetime, no_firetime, rn, total_no_firetime

real*4 :: ran3

integer*4, parameter :: daysXyear=365, years=10000., run=years*daysXyear
real*8, parameter :: time_unit=1.0d0, day2year=1.0d0/daysXyear, time_step=time_unit*day2year
real*8, parameter :: t2=time_step/2.0d0, t6=time_step/6.0d0, eps=0.001d0, min_firetime=2., max_firetime=500.
real*8, parameter :: Cmax=100., Cmin=0.01, eta=0.9, alfa=(N*Cmin-Cmax)/(N-1)
real*8, parameter :: beta = (Cmax-Cmin)/(N-1), gamma = (Cmax-Cmin)/(N*eta)

real, allocatable :: nums(:)
real :: temp1(90), temp2(90), temp3(91)
integer :: index