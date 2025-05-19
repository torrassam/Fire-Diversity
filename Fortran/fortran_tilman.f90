program tilman

use time_ftcs, only: timestamp 

implicit none

include "fortran_tilman.h"

write(string_n, '(i0)') N  
filename = '../results2/fixed_points_'//trim(string_n)//'_'//trim(timestamp())//'.txt'
open(72, file=filename)
filename = '../results2/coefficients_'//trim(string_n)//'_'//trim(timestamp())//'.txt'
open(73, file=filename)
filename = '../results2/firetimes_'//trim(string_n)//'_'//trim(timestamp())//'.txt'
open(81, file=filename, position='append')

call cpu_time(start_time)

write(*,*) "Performing simulation"
write(*,*) alfa, beta, gamma

do comm_gen=1,P

	call comm_gen_new(C,M,R,L,comm_gen)

	write(73,'(I4)', advance='no') comm_gen
	do i=1,N
		write(73,'(4F14.5)', advance='no') C(i), M(i), R(i), L(i)
	end do
	write(73,*)

	do init_cond=1,N+2

		! write(*,*) init_cond, ' / ', N+2

		call initial_conditions(X, init_cond)

		no_firetime=0.
		total_no_firetime=0.
		number_fires=0
		Xtot = 0.

		do i = 1,run
			
			call RK4(X,dX,Xout,C,M,R,L)

			X = Xout

			inv_firetime = 0.
			do j=1,N
				inv_firetime = inv_firetime+L(j)*X(j)
			end do
			ave_firetime = 1./(inv_firetime+eps)*daysXyear

			call date_and_time(VALUES=vals)

			dummy = vals(8)
			rn = ran3(dummy)

			if ((no_firetime>=min_firetime).and.(rn<=1./ave_firetime)) then
				
				if (i.gt.(run*0.5)) then
					!total_no_firetime = total_no_firetime+no_firetime
					number_fires = number_fires+1
				endif

				no_firetime = 0.
				
				do j=1,N
					Xout(j) = X(j)*R(j)
				end do
				
			else

				no_firetime = no_firetime + day2year ! Giorni senza fuochi

			end if
			
			X = Xout

			if (i.gt.(run*0.5)) then
				Xtot = Xtot+X/(run*0.5)
			end if

		enddo
		
		write(81,*) comm_gen, init_cond, (run*0.5/daysXyear)/number_fires

		write(72,*) comm_gen, init_cond, Xtot
		
	enddo

enddo

call cpu_time(end_time)

elapsed_time = end_time-start_time

write(*,*) N, P, years
write(*,*) elapsed_time

close(72)
close(73)
close(81)

end program tilman

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine derivs(X,dX,C,M,R,L)

implicit none

include "fortran_tilman.h"

do i1=1,N
	
	dX(i1)=C(i1)*X(i1)*(1.-sum(X(1:i1)))-M(i1)*X(i1)-X(i1)*dot_product(C(1:(i1-1)),X(1:(i1-1)))
	
enddo

return

end subroutine derivs

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine initial_conditions(X, init_cond)

implicit none

include "fortran_tilman.h"

if (init_cond==N+1) then
	X = 0.01

else if (init_cond==N+2) then
	X = 1./(N+1.)

else
	X = 0.01
	X(init_cond) = 0.91-0.01*N

end if

return

end subroutine initial_conditions

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine comm_gen_new(C,M,R,L,comm_gen)

implicit none

real*8, parameter :: c_coef=0.03023, c_inter=-0.01547, m_coef=0.00881, m_inter=-0.00382
real*8, parameter :: c_std=0.03435, m_std=0.01526
real*8 :: j2
real :: rnorm

include "fortran_tilman.h"

do j=1,N

	j2=(5.*j+N-6)/(N-1)

	do while (.true.)
		C(j) = c_coef*j2+c_inter+c_std*rnorm()
		M(j) = m_coef*j2+m_inter+m_std*rnorm()
		if (C(j)>M(j) .and. M(j)>0) then
			exit 
		end if
	end do

	
	call date_and_time(VALUES=vals)
	dummy = vals(8)

	rn = ran3(vals(8))
	R(j) = 0.001+0.999*rn

	rn = ran3(vals(8))*(log10(max_firetime)-log10(min_firetime))+log10(min_firetime)
	L(j) = 1/10**rn
	
enddo

return

end subroutine comm_gen_new

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine comm_gen_new_bor(C,M,R,L,comm_gen)

implicit none

real*8, parameter :: c_coef=0.0425, c_inter=0.0433, m_coef=-0.0060, m_inter=0.0363
real*8, parameter :: c_std=0.0012, m_std=0.0066 
real*8 :: j2
real :: rnorm

include "fortran_tilman.h"

do j=1,N

	j2=(2.*j+N-3)/(N-1)

	do while (.true.)
		C(j) = c_coef*j2+c_inter+c_std*rnorm()
		M(j) = m_coef*j2+m_inter+m_std*rnorm()
		if (C(j)>M(j) .and. M(j)>0) then
			exit 
		end if
	end do

	
	call date_and_time(VALUES=vals)
	dummy = vals(8)

	rn = ran3(vals(8))
	R(j) = 0.001+0.999*rn

	rn = ran3(vals(8))*(log10(max_firetime)-log10(min_firetime))+log10(min_firetime)
	L(j) = 1/10**rn
	
enddo

return

end subroutine comm_gen_new_bor

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine RK4(v,dv,vout,C,M,R,L)

implicit none

include "fortran_tilman.h"

call derivs(v,dv,C,M,R,L)
vt=v+t2*dv

call derivs(vt,dvt,C,M,R,L)
vt=v+t2*dvt

call derivs(vt,dvm,C,M,R,L) 
vt=v+time_step*dvm
dvm=dvt+dvm

call derivs(vt,dvt,C,M,R,L)
vout=v+t6*(dv+dvt+2.*dvm)

return

end subroutine RK4

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!


FUNCTION RAN3(IDUM)
      ! RANDOM NUMBER GENERATOR - FROM NUMERICAL RECIPES IN FORTRAN
      
      SAVE
      PARAMETER (MBIG=1000000000,MSEED=161843398,MZ=0,FAC=1.E-9)
      DIMENSION MA(55)
      DATA IFF /0/
      IF(IDUM.LT.0.OR.IFF.EQ.0)THEN
        IFF=1
        MJ=MSEED-IABS(IDUM)
        MJ=MOD(MJ,MBIG)
        MA(55)=MJ
        MK=1
        DO 11 I=1,54
          II=MOD(21*I,55)
          MA(II)=MK
          MK=MJ-MK
          IF(MK.LT.MZ)MK=MK+MBIG
          MJ=MA(II)
11      CONTINUE
        DO 13 K=1,4
          DO 12 I=1,55
            MA(I)=MA(I)-MA(1+MOD(I+30,55))
            IF(MA(I).LT.MZ)MA(I)=MA(I)+MBIG
12        CONTINUE
13      CONTINUE
        INEXT=0
        INEXTP=31
        IDUM=1
      ENDIF
      INEXT=INEXT+1
      IF(INEXT.EQ.56)INEXT=1
      INEXTP=INEXTP+1
      IF(INEXTP.EQ.56)INEXTP=1
      MJ=MA(INEXT)-MA(INEXTP)
      IF(MJ.LT.MZ)MJ=MJ+MBIG
      MA(INEXT)=MJ
      RAN3=MJ*FAC
      RETURN
      END

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

FUNCTION rnorm() RESULT( fn_val )

!   Generate a random normal deviate using the polar method.
!   Reference: Marsaglia,G. & Bray,T.A. 'A convenient method for generating
!              normal variables', Siam Rev., vol.6, 260-264, 1964.
!   https://wp.csiro.au/alanmiller/random.html

IMPLICIT NONE
REAL  :: fn_val

! Local variables

REAL            :: u, sum
REAL, SAVE      :: v, sln
LOGICAL, SAVE   :: second = .FALSE.
REAL, PARAMETER :: one = 1.0, vsmall = TINY( one )

IF (second) THEN
! If second, use the second random number generated on last call

  second = .false.
  fn_val = v*sln

ELSE
! First call; generate a pair of random normals

  second = .true.
  DO
    CALL RANDOM_NUMBER( u )
    CALL RANDOM_NUMBER( v )
    u = SCALE( u, 1 ) - one
    v = SCALE( v, 1 ) - one
    sum = u*u + v*v + vsmall         ! vsmall added to prevent LOG(zero) / zero
    IF(sum < one) EXIT
  END DO
  sln = SQRT(- SCALE( LOG(sum), 1 ) / sum)
  fn_val = u*sln
END IF

RETURN
END FUNCTION rnorm

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module time_ftcs

contains
  function timestamp() result(str)
    implicit none
    character(len=20)                     :: str
    integer                               :: values(8)
    character(len=4)                      :: year
    character(len=2)                      :: month
    character(len=2)                      :: day, hour, minute, second
    character(len=5)                      :: zone

    ! Get current time
    call date_and_time(VALUES=values, ZONE=zone)  
    write(year,'(i4.4)')    values(1)
    write(month,'(i2.2)')   values(2)
    write(day,'(i2.2)')     values(3)
    write(hour,'(i2.2)')    values(5)
    write(minute,'(i2.2)')  values(6)
    write(second,'(i2.2)')  values(7)

    str = year//'-'//month//'-'//day//'_'&
          //hour//':'//minute//':'//second
  end function timestamp
end module