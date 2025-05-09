/*
N E W		R O U T I N E		F O R 		F U N C T I O N 	K_D
by Tri
*/

#include "MPDPFT_HEADER_ONEPEXDFT.h"
#include "Plugin_KD.h"
#include "mpDPFT.h"
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <chrono>
#include <functional>
#include <limits>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <vector>
#include <gsl/gsl_sf_airy.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_fermi_dirac.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_zeta.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;

const double PI = 3.14159265358979323846;
double twoPI = 6.283185307179586;
double PIhalf = 1.5707963267948966;


struct Params {int D; double A; double Bcube;\
            double thirdBcube; double s0p; double s1ref; double s2ref;};
	    
struct ParamHO {double mu; double r;double reltolx;};

//function phi(s), taking s, A, B^3/3 as para.
double phifunc(double s, double A, double thirdBcube) {
	double ssq = s * s;
	return (((thirdBcube * ssq + A) * ssq + 0.25) / s);
}

//evaluations of phi(s) and phi'(s) needed for Newton-Raphson method,
//taking s, A, B^3, B^3/3 as para., storing phi at &phi and phi' at &derphi
void newtonfunc(double s, double A, double Bcube, double thirdBcube, double &phi, double &derphi) {
	double ssq = s * s;
	phi = ((thirdBcube * ssq + A) * ssq + 0.25) / s;
	derphi = A + Bcube * ssq - 0.25 / ssq;
}

//solver for s1 & s2, taking x = phi, A, B^3, B^3/3, s0 = s0p, and reference points s1ref & s2ref as para.,
//storing s1 at &s1 and s2 at &s2
void s12(double x, double A, double Bcube, double thirdBcube,\
			double s0p, double &s1, double &s2, double s1ref = 0., double s2ref = 0.) {
	double phi,derphi;
	double dx,temp;
	int i;
	//find reference points (if not supplied by user) for Newton-Raphson
    if (s1ref == 0.) {
        s1 = s0p;
        s2 = s0p;
        do {
            s1 *= 0.5;
            phi = phifunc(s1,A,thirdBcube);
        }
        while (phi < x);
		
        double delta(1.);
        do {
            delta *= 2.;
            s2 += delta;
            phi = phifunc(s2,A,thirdBcube);
        }
        while (phi < x);
    }
    else {
        s1 = s1ref;
        s2 = s2ref;
    }
    
    
    
	//Newton-Raphson
	for (i=0;i<100;i++) {
		newtonfunc(s1,A,Bcube,thirdBcube,phi,derphi);
		dx = (phi-x)/derphi;
		temp = s1;
		s1 -= dx;
		if (s1 == temp) break; //if dx is too small to make a visible change in s1 within machine precision, stop
		if (ABS(dx) < 1e-17) break; //abstols1s2 reached, stop
	}
	
	for (i=0;i<100;i++)  {
		newtonfunc(s2,A,Bcube,thirdBcube,phi,derphi);
		dx = (phi-x)/derphi;
		temp = s2;
		s2 -= dx;
		if (s2 == temp) break; //if dx is too small to make a visible change in s1 within machine precision, stop
		if (ABS(dx) < 1e-17) break; //abstols1s2 reached, stop
	}
	
}


double sDiff(double x, void * p) {
        struct Params * params  = (struct Params *)p;
        int D                   = (params->D);
        double A                = (params->A); 
        double Bcube            = (params->Bcube);
        double thirdBcube       = (params->thirdBcube);
        double s0p              = (params->s0p);
        double s1ref            = (params->s1ref);
        double s2ref            = (params->s2ref);
        
        double s1, s2;
        double Dd = (double)D;
        s12(x, A, Bcube, thirdBcube, s0p, s1, s2, s1ref, s2ref);
        double aux1 = pow(0.25 / s1, Dd); //aux1 = 1 / (4 * s1)^D
        double aux2 = pow(0.25 / s2, Dd); //aux2 = 1 / (4 * s2)^D
        return aux2 - aux1;
}

double gx(double x, void * p) {
        struct Params * params  = (struct Params *)p;
        int D                   = (params->D);
        double A                = (params->A); 
        double Bcube            = (params->Bcube);
        double thirdBcube       = (params->thirdBcube);
        double s0p              = (params->s0p);
        double s1ref            = (params->s1ref);
        double s2ref            = (params->s2ref);
        
        double s1, s2;
        double Dd = (double)D;
        s12(x, A, Bcube, thirdBcube, s0p, s1, s2, s1ref, s2ref);
        double aux1, aux2, s1sq;
        s1sq = s1 * s1;
        aux1 = (A + thirdBcube * s1sq) * s1; //aux1 = A * s1 + B^3 * s1 / 3
        aux2 = pow(0.25 / s2, Dd); //aux2 = 1 / (4 * s2)^D
        switch (D) {
            case 1: 
                return aux1 + aux2;
				break;
			
			case 2:
				return (aux2 + pow(aux1,2.) + 0.5 * thirdBcube * s1sq);
				break;
				
			case 3:
				return (aux2 + pow(aux1,3.) \
							+ 0.25 * aux1 * Bcube * s1sq \
							+ 0.0625 * Bcube * s1);
				break;
			default:
                return 0.;
				break;
		}
}

//calculate phi0 and return s0 at the same time
double phi0(double A, double B, double &s0p) {
	double A2 = A * A;
	double B3 = B * B * B;
	//if (A >= 0) {
	if (A>1.0e-14) {//MIT
		s0p = sqrt(0.5 / (sqrt(A2 + B3) + A));
	}
	else {
		s0p = sqrt(0.5 * (sqrt(A2 + B3) - A) / B3);
	}
	return (2. * A * s0p + 4. * B3 * pow(s0p,3.) / 3.);
}

//MIT
// double phi0(double A, double B, double &s0) {
// 	double A2 = A * A, B3 = B * B * B;
// 	if(B3>1.0e-16 && A2>1.0e-16*B3) s0 = sqrt(0.5/(A*(1.+sqrt(1.+B3/A2))));
// 	else if(B3>1.0e-16) s0 = sqrt(0.5/sqrt(B3));
// 	else{
// 	  s0 = sqrt(0.5/(Sign(A)*1.0e-32*2.));
// 	  return Sign(A)*1.0e-32*s0+1.0e-16*s0*s0*s0/3.+1./(4.*s0);
// 	}
// 	
// 	return A*s0+B3*s0*s0*s0/3.+1./(4.*s0);
// }

//second derivative of phi
double phipp(double s, double B) {
	double s3 = pow(s,3.);
	return (2 * pow(B,3.) * s + 0.5 / s3);
} 

//estimate for KD; when A<0 and 3 * A^2 > B^3, use asymptotics for large negative A;
//else, use asymptotics for large B
double KDapr(int D, double A, double B) {
	double res, Dd;
	Dd = (double)D;
	if ((pow(A,2.) > pow(B,3.)/3) && A < 0.) {
		double Ap = ABS(A);
		double arg = pow(Ap / B, 1.5) / 1.5 \
					+ 0.25 * PI + Dd * PIhalf \
					- 0.25 * pow(B, 1.5) / sqrt(Ap);
		res = pow(0.5 * B, Dd) * (cos(arg) / (pow(Ap / B, 0.75 + 0.5 * Dd))\
				+  (41. / 48. + Dd * (0.25 * Dd + 1.)) * sin(arg)\
						/ (pow(Ap / B, 2.25 + 0.5 * Dd))) / sqrt(PI);
	}
	else {
		double s0p;
		double phi0p = phi0(A, B, s0p);
		double pp = phipp(s0p,B);
		res = (2. / pow(2. * s0p, Dd + 1.)) * sqrt(2. / (PI * pp))\
				* sin(phi0p - Dd * PIhalf + 0.25 * PI);
	}
	return res;
}

//Alex's results; so far: Tri's asymps for A<0 is better, but Alex's asymps. for B>0 is better.
double KDasymp(int D, double A, double B) {
    double res;
    double A2 = A*A;
    double B3 = B*B*B;
    
    if (A < 0 and 3. * A2 > B3) {
        double Ap = ABS(A);
        double arg = (2. * pow(Ap/B,1.5) / 3. - 0.25 * pow(B3/Ap,0.5));
        double c1 = 1. - B3 / (16. * A2);
        double c2 = 0.25 * pow(B/Ap,1.5);
        double c3 = pow(twoPI,0.5);
        switch (D) {
            case 1:
                {res = -0.5 * pow(B,9./4.)/(c3 * pow(Ap,5./4.))*(cos(arg)*(c1-c2)+sin(arg)*(c1+c2));
                break;}
            case 2:
                {res = 0.25*pow(B,15./4.)/(c3*pow(Ap,7./4.))*(-cos(arg)*(c1+4.*c2)+sin(arg)*(c1-4.*c2));
                break;}
            case 3:
                {res = pow(B,21./4.)/(8.*c3*pow(Ap,9./4.))*(cos(arg)*(c1-9.*c2)+sin(arg)*(c1+9.*c2));
                break;}
            default:
                {res = 0.;
                break;}
        }
        return res;
    }
    else {
        double arg = A /(sqrt(2.)*pow(B,0.75))+sqrt(2.)*pow(B,0.75)/3.;
        switch (D) {
            case 1:
                {res = -pow(B,3./8.)/(sqrt(PI)*pow(2.,0.75))*(cos(arg)*(1-sqrt(2.)/pow(B,0.75))-sin(arg)*(1+sqrt(2.)/pow(B,0.75)));
                break;}
            case 2:
                {res = -pow(B,9./8.)/(sqrt(PI)*pow(2.,5./4.))*(cos(arg)*(1.+5./(sqrt(2.)*pow(B,0.75)))+sin(arg)*(1.-5./(sqrt(2.)*pow(B,0.75))));
                break;}
            case 3:
                {res = pow(B,15./8.)/(sqrt(PI)*pow(2.,7./4.))*(cos(arg)*(1.-9./(sqrt(2.)*pow(B,0.75)))-sin(arg)*(1.+9./(sqrt(2.)*pow(B,0.75))));
                break;}
            default:
                {res = 0.;
                break;}
        }
        return res;
    }
    
}


//MIT
void Gets1s2(double phi, double s0, double A, double Bcube, double thirdBcube, double &s1, double &s2){
  s1 = 0.5*s0;
  while(phifunc(s1, A, thirdBcube)<0.) s1 *= 0.5;
  double deltas1 = s1;
  while(abs(phifunc(s1, A, thirdBcube)-phi)>1.0e-6 && deltas1>1.0e-12){
    deltas1 *= 0.5;
    if(phifunc(s1, A, thirdBcube)>phi) s1 += deltas1; else s1 -= deltas1;
  }
  s2 = 2.*s0;
  while(phifunc(s2, A, thirdBcube)<0.) s2 *= 2.;
  double deltas2 = 0.5*(s2-s0);
  while(abs(phifunc(s2, A, thirdBcube)-phi)>1.0e-6 && deltas2>1.0e-12){
    deltas2 *= 0.5;
    if(phifunc(s2, A, thirdBcube)<phi) s2 += deltas2; else s2 -= deltas2;
  }  
}

double KD_Contour(int D, double A, double B, KDintegrationParams &KDip){
    KDip.IntegrationArraySize = 10000;
    KDip.minB = 0.01;
    return KD_contour(D, A, B, KDip);
}

double KD(int D, double A, double B, double &abserr, double reltolx, KDintegrationParams &KDip) {
  
  //MIT20200914: introduce additional USERINPUT accuracy variables:
  int IntegrationArraySize = KDip.IntegrationArraySize;//default=10
  double KDthreshold = KDip.KDthreshold;//absolute-accuracy threshold; default=1.0e-7
  int iMAX = KDip.iMAX;///maximum number of loops for reaching target accuracy; default=100
  int kQAWO = KDip.kQAWO;//range of QAWO integration; default=100
  double minB = KDip.minB;//return approximation if B<minB; default: default=1.0e-4
  double maxK = KDip.maxK;//return KDapr if floor(ABS(phi0p)/(2.*PI*k))>maxK; default=1.0e+6
  int stepM = KDip.stepM;//integration range for first estimation; default=1000

  //overkill:
  // int IntegrationArraySize = 1000*KDip.IntegrationArraySize;//default=10
  // double KDthreshold = 0.001*KDip.KDthreshold;//absolute-accuracy threshold; default=1.0e-7
  // int iMAX = 100*KDip.iMAX;///maximum number of loops for reaching target accuracy; default=100
  // int kQAWO = 10*KDip.kQAWO;//range of QAWO integration; default=100
  // double minB = KDip.minB;//return approximation if B<minB; default: default=1.0e-4
  // double maxK = 100.*KDip.maxK;//return KDapr if floor(ABS(phi0p)/(2.*PI*k))>maxK; default=1.0e+6
  // int stepM = 10*KDip.stepM;//integration range for first estimation; default=1000

  	//Alex' and Michael's contour integral


  

    if (B < minB) {//B too small, approximate by sqrt(A_+)^D * J_D(sqrt(A_+))
        abserr = -1.;
		double arg = sqrt(max(A,0.));
		return POW(arg,D) * gsl_sf_bessel_Jn(D, arg);
	}
    else {
        abserr = 0.;
        // set up parameters
        double Asq = A * A;
        double Bcube = B * B * B;
        double thirdBcube = Bcube / 3.;
        double phi0p, s0p;
        phi0p = phi0(A, B, s0p);
        struct Params params = {D,A,Bcube,thirdBcube,s0p,0.,0.};
        
        // set up workspace
        gsl_set_error_handler_off();
        gsl_integration_workspace * w = gsl_integration_workspace_alloc(IntegrationArraySize);
        //vector<int> tn = GetNestedThreadNumbers(primeFactors((int)omp_get_max_threads(),0), 0);
        //int num_inner_threads = tn[1];
        vector<gsl_integration_workspace*> W((int)omp_get_max_threads());
        for (int i = 0; i < W.size(); ++i) W[i] = gsl_integration_workspace_alloc(IntegrationArraySize);
        gsl_integration_qawo_table * wf;
        gsl_integration_qawo_table * wf2;
        gsl_function F;

        vector<double> ADD(0);
        vector<double> ADDERR(0);
        
        double res(0.),resp(0.),add(0.),adderr(0.);
        double s1ref,s2ref;
        int K;
        int counter = 0;
        double dd = twoPI*(double)kQAWO;
        double epsabs = 1e-16;// for QAWO
        double epsrel = 0.;
        double lim1 = max(0.,phi0p);
        int M = (int)ceil(lim1/twoPI);

        if(D==1 || D==3){
            wf = gsl_integration_qawo_table_alloc(1.,1.,GSL_INTEG_SINE,IntegrationArraySize);
            wf2 = gsl_integration_qawo_table_alloc(1.,1.,GSL_INTEG_SINE,IntegrationArraySize);
        }
        else if(D==2){
            wf = gsl_integration_qawo_table_alloc(1.,1.,GSL_INTEG_COSINE,IntegrationArraySize);
            wf2 = gsl_integration_qawo_table_alloc(1.,1.,GSL_INTEG_COSINE,IntegrationArraySize);
        }

        gsl_integration_qawo_table_set_length(wf,dd);

        if (A < 0 && Asq > thirdBcube) {
            double alpha = 0.5 * acosh(ABS(A)/sqrt(thirdBcube));
            s1ref = exp(-alpha)/(sqrt(2.)*pow(thirdBcube,0.25));
            s2ref = exp(alpha)/(sqrt(2.)*pow(thirdBcube,0.25));

            // set up workspace
            params.s1ref = s1ref;
            params.s2ref = s2ref;
            F.function = &sDiff;
            F.params = &params;

            // LOOP FROM phi0 < 0 to 0.
            double testK = ABS(phi0p)/dd;
            if (testK > maxK) {
                abserr = -1.;
                return KDapr(D,A,B);
            }
            if(KDip.contourQ) return KD_Contour(D, A, B, KDip);
            K = (int)(ABS(phi0p)/dd);
            int iMax = max(0,K);
            if(K>0){
                ADD.resize(K);
                ADDERR.resize(K);
            }
            #pragma omp parallel num_threads(KDip.num_inner_threads)
            {
                #pragma omp for schedule(dynamic) reduction(+: res)
               	for (int i=0;i<=iMax;i++){
                    if(i==0 && dd*(double)K + phi0p < 0.){
                        gsl_integration_qawo_table_set_length(wf2,-(dd*(double)K+phi0p));
                        int check = gsl_integration_qawo(&F,phi0p,epsabs,epsrel,IntegrationArraySize,w,wf2,&add,&adderr);
                        res += add;
                    }
                    else if(i>0 && K>0){
                   		int check = gsl_integration_qawo(&F,-dd*(double)i,epsabs,epsrel,IntegrationArraySize,W[omp_get_thread_num()],wf,&ADD[i-1],&ADDERR[i-1]);
                   		res += ADD[i-1];
                    }
               	}
            }

        }
        else{
            if(KDip.contourQ) return KD_Contour(D, A, B, KDip);
            if(D==1) res = sin(phi0p) - phi0p * cos(phi0p);
            else if(D==2) res = (-2.*(sin(phi0p) - phi0p * cos(phi0p))+sin(phi0p)*(phi0p*phi0p-0.5*A));
            else if(D==3) res = (-6.*(sin(phi0p) - phi0p * cos(phi0p))+3.*sin(phi0p)*(phi0p*phi0p-0.25*A)-cos(phi0p)*(pow(phi0p,3.)-0.75*A*phi0p));
        }

        // LOOP FROM 0 or phi0 to LIM2
        // FIRST ESTIMATION
        M += stepM;
        double lim2;
        if(D==1 || D==3) lim2 = (double)M*twoPI-PIhalf;
        else if(D==2) lim2 = (double)M*twoPI;

        // set parameters
        s12(lim2,A,Bcube,thirdBcube,s0p,s1ref,s2ref);
        params.s1ref = s1ref;
        params.s2ref = s2ref;
        F.function = &gx;
        F.params = &params;
        double testK = (lim2-lim1)/dd;
        if(testK<0. || testK>2147483646.) return res;
        K = (int)testK;
        if(K>0){
            ADD.resize(K);
            ADDERR.resize(K);
        }
        #pragma omp parallel num_threads(KDip.num_inner_threads)
        {
            #pragma omp for schedule(dynamic) reduction(+: res)
            for (int i=-1;i<K;i++){
                if(i==-1 && lim1+(double)K*dd < lim2){
                    gsl_integration_qawo_table_set_length(wf2,lim2-(lim1+(double)K*dd));
                    int check = gsl_integration_qawo(&F,lim1+(double)K*dd,epsabs,epsrel,IntegrationArraySize,w,wf2,&add,&adderr);
                    res += add;
                }
                else if(i>=0 && K>0){
                	int check = gsl_integration_qawo(&F,lim1+(double)i*dd,epsabs,epsrel,IntegrationArraySize,W[omp_get_thread_num()],wf,&ADD[i],&ADDERR[i]);
                	res += ADD[i];
                }
            }
        }

        resp = res;

        // LOOP TO IMPROVE ON THIS ESTIMATE
        for(int i=1;i<iMAX;i++){
            if(M>2147483646/2) break;
            lim1 = lim2;
            M *= 2;
            if(D==1 || D==3) lim2 = (double)M*twoPI-PIhalf;
            else if(D==2) lim2 = (double)M*twoPI;
            s12(lim2,A,Bcube,thirdBcube,s0p,s1ref,s2ref);
            params.s1ref = s1ref;
            params.s2ref = s2ref;
            F.function = &gx;
            F.params = &params;
            double testK = (lim2-lim1)/dd;
        	if(testK<0. || testK>2147483646.) return res;
        	K = (int)testK;

            // UPDATE RESULT
            if(K>0){
            	ADD.resize(K);
            	ADDERR.resize(K);
            }
            #pragma omp parallel num_threads(KDip.num_inner_threads)
            {
                #pragma omp for schedule(dynamic) reduction(+: res)
                for (int j=-1;j<K;j++){
                    if(j==-1 && lim1+(double)K*dd < lim2){
                        gsl_integration_qawo_table_set_length(wf2,lim2-(lim1+(double)K*dd));
                        int check = gsl_integration_qawo(&F,lim1+(double)K*dd,epsabs,epsrel,IntegrationArraySize,w,wf2,&add,&adderr);
                        res += add;
                    }
                    else if(j>=0 && K>0){
                    	int check = gsl_integration_qawo(&F,lim1+(double)j*dd,epsabs,epsrel,IntegrationArraySize,W[omp_get_thread_num()],wf,&ADD[j],&ADDERR[j]);
                    	res += ADD[j];
                    }
                }
            }

            if (ABS(resp - res) < 0.5*reltolx*ABS(res+resp) || ABS(res) < PIhalf/**Kmax*/*KDthreshold) {counter++;}
            if (counter == 2) {break;}
            resp = res;
        }
        if(D==1) res /= PIhalf;
        else if(D==2) res /= -PIhalf;
        else if(D==3) res *= -8. / (3. * PI);

        //printf ("A = %.10f, B = %.10f, KD = %.10f\n",A,B,res);
        gsl_integration_qawo_table_free(wf);
        gsl_integration_qawo_table_free(wf2);
        gsl_integration_workspace_free(w);
        for (int i = 0; i < W.size(); ++i) gsl_integration_workspace_free(W[i]);
        return res;
        
    }
}



//BEGIN Michael Tsesmelis code


struct params_t {
    double A;
    double B;
    int D;
    double gamma;
    double eta;
    double g2, g3, g4, g6, e2, A2, B3, B6;
    double cosim, sinim;
    int calls;
};

double dvdu(double u, double v, double A, double B, double gamma){
    //params_t *p = (params_t *)params;
    //double A = p->A;
    //double B = p->B;
    //double gamma = p->gamma;

    double g = 4*pow(B, 3)*pow(u, 4)*pow(gamma, 4);

    double p0 = pow(u, 2)*(-1+4*A*pow(u,2)*pow(gamma, 2) + g);
    double p1 = pow(v, 2)*(1 + 8*A*pow(u, 2)*pow(gamma, 2) + g);
    double p2 = pow(v, 4)*4*pow(gamma, 2)*(A - pow(B, 3)*pow(u, 2)*pow(gamma, 2));
    double p3 = 4*pow(B, 3)*pow(gamma, 4)*pow(v, 6);

    double numerator = p0+p1+p2-p3;
    double denominator = 2*u*v*(1+g+4*pow(B, 3)*pow(gamma, 4)*pow(v, 2)*(2*pow(u, 2)+pow(v, 2)));
    double val = numerator / denominator;
    //printf("dvdu: %.10f\n", val);

    return val;
}

bool is_close(double a, double b, double atol = 1.e-8) {
    return std::abs(a - b) <= atol;
}


double fcontour(double u, void *params){
    params_t *p = (params_t *)params;  // Cast params to the correct type

    // Access parameters
    double A = p->A;
    double B = p->B;
    int D = p->D;
    double gamma = p->gamma;
    double eta = p->eta;
    double g2 = p->g2;
    double g3 = p->g3;
    double g4 = p->g4;
    double g6 = p->g6;
    double e2 = p->e2;
    double A2 = p->A2;
    double B3 = p->B3;
    double B6 = p->B6;
    double cosim = p->cosim;
    double sinim = p->sinim;
    //p->calls++;

    int coeff = (u < 1) ? -1 : 1;

    double v;
    //std::cout << B3 << " " << B6 << std::endl;
    //printf("gamma: %.10f\n", gamma);
    //printf("eta: %.10f\n", eta);
    //printf("A: %.10f\n", A);
    //printf("B: %.10f\n", B);
    //printf("coeff: %.10f\n", coeff);


    // Determine v based on contour
    double p0, p1, p2, p3, p4, p5, p6, p7, p8;

    double u2 = pow(u, 2);
    double u3 = pow(u, 3);
    double u4 = pow(u, 4);
    double u6 = pow(u, 6);
    double A0, B0;

    if (u==1){
        v=0;
    } else {
        p0 = 3*A*u*g2;
        p1 = 2*B3*u3*g4;
        p2 = 3*gamma*eta;
        p3 = 9*(A2+B3)*u2*g2;
        p4 = 24*A*B3*u4*g4;
        p5 = 16*B6*u6*g6;
        p6 = 6*u*gamma*(3*A + 4*B3*u2*g2 )*eta;
        p7 = 9*e2;
        p8 = 6*B3*u*g4;

        A0 = p0 - p1 - p2;
        B0 = sqrt( g2 * (p3 + p4 + p5 - p6 + p7));

        if (is_close(A0, -B0, 1e-12) && (0.9 <= u <= 1.1))
            v = u-1;
        else
            v = coeff * sqrt( (A0 + B0) / (p8));

    }
    double v2 = pow(v, 2);
    double v3 = pow(v, 3);
    double v4 = pow(v, 4);
    double v6 = pow(v, 6);

    //printf("u: %.10f\n", u);
    //printf("v: %.10f\n", v);

    // Plug in (u, v) into real part of integral
    //std::complex<double> t(u, v); 
    double Reexponent = -A*gamma*v - g3 * (3*u2*v-v3) * B3/3 + v/(4*gamma*(u2+v2));
    //double Imexponent = A*gamma*u + 1/3*pow(gamma, 3)*(pow(u, 3)-3*u*pow(v,2))*pow(B, 3) + u/(4*gamma*(pow(u, 2)+pow(v, 2)));
    //double Imexponent = A*gamma + pow(gamma, 3)/3 + 1/(4*gamma);


    //std::complex<double> coeff_term1 = - 1.0 / (4*gamma*pow(t, 2));

    // Calculate dv / du term
    double dvdu_term;
    if (u==1 || (is_close(u, v+1, 1e-12))/* && (0.9 <= u <= 1.1))*/ ){
        dvdu_term = 1.;
    } else {
        double g = 4*B3*u4*g4;
        double p00 = u2*(-1 + 4*A*u2*g2 + g);
        double p01 = v2*(1 + 8*A*u2*g2 + g);
        double p02 = v4*4*g2*(A - B3*u2*g2);
        double p03 = 4*B3*g4*v6;

        double numerator = p00+p01+p02-p03;
        double denominator = 2*u*v*(1 + g + 4*B3*g4*v2*(2*u2+v2) );
        dvdu_term = numerator / denominator;
    }
    //std::complex<double> jac = std::complex<double>(1, dvdu_term);
    double re = exp(Reexponent);
    //std::complex<double> im = cosim + std::complex<double>(0.0, 1)*sinim; //std::exp(std::complex<double>(0.0, eta));

    double val, val1, val2;

    //std::complex<double> denom = 2.0 * gamma * std::complex<double>(0, 1) * t;
    //std::complex<double> coeff_term =  gamma * 1.0 / pow(denom, D+1);
    //printf("coeff: %.10f\n", real(coeff_term));
    
    //printf("coeff analysis: %.10f\n", - gamma * real(std::complex<double>((u2-v2)/ 4*g2*pow((u2+v2), 2), 2*u*v/ 4*g2*pow((u2+v2), 2))) );
    //val = re * real(coeff_term * jac * im);

    if (D==1){
        val1 = (u2-v2) * (cosim - dvdu_term * sinim); // Re
        val2 = 2*u*v * (sinim + dvdu_term * cosim); // Im * Im
        val = - re * (val1 + val2) / (4*gamma*pow((u2+v2), 2));
    } else if (D==2){
        val1 = (3*u2*v-v3) * (cosim - dvdu_term*sinim);
        val2 =  - (u3 - 3*u*v2) * (sinim + dvdu_term * cosim);
        val = re * (val1 + val2) / (8*g2*pow((u2+v2), 3));
    } else if (D==3){
        val1 = (u4-6*u2*v2+v4) * (cosim - dvdu_term*sinim);
        val2 =  - (-4*u3*v+4*u*v3) * (sinim + dvdu_term * cosim);
        val = re * (val1 + val2) / (16*g3*pow((u2+v2), 4));
    }

    //printf("imexponent: %.10f\n", eta);

    //printf("coeff: %.10f\n", real(coeff_term * re * jac * im));
    //printf("u: %.10f\n", u);
    //printf("val1: %.10f\n", cosim);
    //printf("val2: %.10f\n", dvdu_term);

    //printf("val: %.10f\n", val);

    return val;
}


double fcontour_gaussian(double u, void *params){
    params_t *p = (params_t *)params;  // Cast params to the correct type

    // Access parameters
    double A = p->A;
    double B = p->B;
    int D = p->D;
    double gamma = p->gamma;
    double eta = p->eta;
    double g2 = p->g2;
    double g3 = p->g3;
    double g4 = p->g4;
    double g6 = p->g6;
    double e2 = p->e2;
    double A2 = p->A2;
    double B3 = p->B3;
    double B6 = p->B6;
    double cosim = p->cosim;
    double sinim = p->sinim;

    double v;

    int coeff = (u < 1) ? -1 : 1;

    // Determine v based on contour
    double p0, p1, p2, p3, p4, p5, p6, p7, p8;

    double u2 = pow(u, 2);
    double u3 = pow(u, 3);
    double u4 = pow(u, 4);
    double u6 = pow(u, 6);
    double A0, B0;

    if (u==1){
        v=0;
    } else {
        p0 = 3*A*u*g2;
        p1 = 2*B3*u3*g4;
        p2 = 3*gamma*eta;
        p3 = 9*(A2+B3)*u2*g2;
        p4 = 24*A*B3*u4*g4;
        p5 = 16*B6*u6*g6;
        p6 = 6*u*gamma*(3*A + 4*B3*u2*g2 )*eta;
        p7 = 9*e2;
        p8 = 6*B3*u*g4;

        A0 = p0 - p1 - p2;
        B0 = sqrt( g2 * (p3 + p4 + p5 - p6 + p7));

        if (is_close(A0, -B0))
            v = u-1;
        else
            v = coeff * sqrt( (A0 + B0) / (p8));
    }

    double v2 = pow(v, 2);
    double v3 = pow(v, 3);
    double v4 = pow(v, 4);
    double v6 = pow(v, 6);

    // Calculate dv / du term
    double dvdu;
    if (u==1 || is_close(u, v+1) ){
        dvdu = 1.;
    } else {
        double g = 4*B3*u4*g4;
        double p00 = u2*(-1 + 4*A*u2*g2 + g);
        double p01 = v2*(1 + 8*A*u2*g2 + g);
        double p02 = v4*4*g2*(A - B3*u2*g2);
        double p03 = 4*B3*g4*v6;

        double numerator = p00+p01+p02-p03;
        double denominator = 2*u*v*(1 + g + 4*B3*g4*v2*(2*u2+v2) );
        dvdu = numerator / denominator;
    }

    double dvdu2 = pow(dvdu, 2);
    double ddvduu;

    double t01, t02, t03, t04;
    double u2v2 = pow(u, 2) + pow(v, 2);
    double u2v2_2 = pow(u2v2, 2);
    double u2v2_3 = pow(u2v2, 3);

    t01 = 2*v*dvdu* (4*B3*g4 + (-3*u2 + v2)/u2v2_3);
    t02 = - (4*u*v2*dvdu2) / u2v2_3;
    t03 = -4*B3*g4 + (-u2 + 3*v2) / u2v2_3;
    t04 = 4*B3*g4 + 1 / (u2v2_2);
    ddvduu = (t01 + t02 + u*(t03 + dvdu2 * t04)) / (u*v*t04);

    //cout << (u*v*t04) << endl;
    //cout << "u: " << u << " v: " << v << endl;
    //cout << t01 << " " << t02 << " " << t03 << " " << t04 << endl;
    //cout << "ddvduu: " << ddvduu << endl; 
    double a, b, c;
    a = -A*gamma*v + v/(4*gamma*(1+v2)) + 1./3*B3*g3*(-3*v+v3);
    b = -A*gamma*dvdu + (-2*v+dvdu-v2*dvdu)/(4*gamma*pow((1+v2), 2)) 
        + B3*g3*(-2*v-dvdu+v2*dvdu);
    c = -1./2*A*gamma*ddvduu + 1./2*B3*g3*(-2*v - 4*dvdu + 2*v*dvdu2 
            - ddvduu + v2*ddvduu) 
        + 1 / (8*gamma*pow((1+v2), 3)) * (6*v-2*v3-4*dvdu+12*v2*dvdu -
            6*v*dvdu2 + 2*v3*dvdu2 + ddvduu - v4*ddvduu);
    
    //cout << a << " " << b << " " << c << endl;
    double re;
    if (c>=0){
        double t1 = -0.25*pow(b, 2) / c;
        double t2 = (b - 2*c/5) / (2*sqrt(c));
        double t3 = (2*b + c) / (4*sqrt(c));
        double t4 = (-0.886227*gsl_sf_erf(t2) + 0.886227*gsl_sf_erf(t3));

        re = 1/sqrt(c) * pow(2.71828, a) * exp(t1) * t4;
    } else {
        double b2 = pow(b, 2);
        double c2 = pow(c, 2);
        double t1 = a - pow(b, 2)/(4*c);
        double t2 = sqrt(-0.1*b - (0.25*b2) / c - 0.01*c);
        double t3 = (0.443113*b2 - 0.97485 *b*c + 0.177245*c2);
        double t4 = 0.5*sqrt( -pow(b-0.2*c, 2) / c);
        double t5 = sqrt(0.1*b - (0.25*b2) / c - 0.01*c);
        double t6 = (-0.443113*b2 + 0.797604*b*c + 0.177245*c2);
        double t7 = (0.1 - 0.5*b/c) * (0.1 + 0.5*b/c) * (b - 2*c) * c2;
        re = exp(t1) * (t2 * t3 * gsl_sf_erf(t4) + t5*t6*gsl_sf_erf(t7)) / t7;
    }

    double val1, val2, result;
    if (D==1){
        val1 = (u2-v2) * (cosim - dvdu * sinim); // Re
        val2 = 2*u*v * (sinim + dvdu * cosim); // Im * Im
        result = - re * (val1 + val2) / (4*gamma*pow((u2+v2), 2));
    } else if (D==2){
        val1 = (3*u2*v-v3) * (cosim - dvdu*sinim);
        val2 =  - (u3 - 3*u*v2) * (sinim + dvdu * cosim);
        result = re * (val1 + val2) / (8*g2*pow((u2+v2), 3));
    } else if (D==3){
        val1 = (u4-6*u2*v2+v4) * (cosim - dvdu*sinim);
        val2 =  - (-4*u3*v+4*u*v3) * (sinim + dvdu * cosim);
        result = re * (val1 + val2) / (16*g3*pow((u2+v2), 4));
    }


    return result;


}



double KD_contour(int D, double A, double B, double eta, double gamma, gsl_integration_workspace *workspace/*, std::vector<double> &fc*/){

    double g2 = pow(gamma, 2);
    double g3 = pow(gamma, 3);
    double g4 = pow(gamma, 4);
    double g6 = pow(gamma, 6);
    double e2 = pow(eta, 2);
    double A2 = pow(A, 2);
    double B2 = pow(B, 2);
    double B3 = pow(B, 3);
    double B6 = pow(B, 6);

    double cosim = cos(eta);
    double sinim = sin(eta);

    params_t params = {A, B, D, gamma, eta, g2, g3, g4, g6, e2, A2, B3, B6, cosim, sinim, 0};

    // First integral from 0 to 1
    double result, error;
    gsl_function F;
    F.function = &fcontour;
    F.params = &params;
    size_t neval;
    //auto start = std::chrono::high_resolution_clock::now();
    //for(int i=0; i<40; i++){
    //    result = fcontour(i/100, &params);
    //}
    //gsl_integration_qag(&F, x_m - w, x_m + w, 1e-6, 1e-6, 100, GSL_INTEG_GAUSS10, w, &result, &error);

    //gsl_integration_qng(&F, 0.01, 100., 1e-2, 1000, &result, &error, &neval);

    if (B < 1e-4){
        double arg = sqrt(max(A, 0.));
        return pow(arg, D) * gsl_sf_bessel_Jn(D, arg);
    }
    double x = sqrt(A2+B2);
    int L = 1;
    double result_gaussian;
    if(A>100000 || B>100000 || (A>1e5 && B < 1.) || (A < -10 && (std::abs(A) > 100*std::abs(B))) || (A<1 && B >= 1e4)){
        cout <<  "Gaussian contour" << endl;
        result = fcontour_gaussian(1.000001, &params);

    } else {
        cout <<  "GSL Integration" << endl;
        gsl_integration_qags(&F, 0.0001, 3., 1.e-4, 1.e-4, 1000, workspace, &result, &error); // take out e^I \eta
    } 
    //auto mid = std::chrono::high_resolution_clock::now();

    double total_result = 2/3.14159 * result;
    //cout << "integration result: " << total_result << " gaussian:" << result_gaussian << endl;

    //std::chrono::duration<double> duration1 = mid - start;

    //std::cout << "Time taken inside loop: " << duration1.count() << " with calls " << neval << std::endl;
    //for(double i=0.1; i<5.0;i+=0.1){
    //    fc.push_back(fcontour(i, &params));
    //}


    return total_result;
}

std::tuple<double, double, double, double> run_contour(int D, double A, double B, std::vector<double> &fc){

    struct KDintegrationParams KDip;
    double gamma;
    if (A > 0 && B < 1e-2)
        gamma = 0.5 * 1./sqrt(A);
    else
        gamma = sqrt( (-A + sqrt(pow(A, 2) + pow(B, 3))) / (2*pow(B, 3)) );
    printf("gamma original %.10f\n", gamma);
    double eta = A*gamma + 1./3. * pow(gamma, 3)*pow(B, 3) + 1./(4*gamma);
    //printf("eta original %.10f\n", eta);

    params_t params = {A, B, D, gamma, eta, 1};
    //double sp = fcontour(0.62626263, &params);

    //double contour_advanced = 1;
    double contour_advanced;
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000);

    //for (int i=1; i<1000; i++){
    auto start = std::chrono::high_resolution_clock::now();

    contour_advanced = KD_contour(D, A, B, eta, gamma, workspace/*, fc*/);
    //}
    auto mid = std::chrono::high_resolution_clock::now();

    gsl_integration_workspace_free(workspace);



    double err = 0.;
    double contour;

    //for (int i=0; i<1000; i++){
    cout << "D: " << D << " A: " << A << " B: " << B << endl;
    contour = KD(D, A, B, err, 0.01, KDip);
    //}
    auto end = std::chrono::high_resolution_clock::now();

    cout << "A: " << A << ", B: " << B << endl;
    printf("Existing KD integration result: %.10f\n", contour);
    printf("Contour KD integration result: %.10f\n", contour_advanced);

    std::chrono::duration<double> duration1 = mid - start;
    std::chrono::duration<double> duration2 = end - mid;

    std::cout << "Time taken for contour: " << duration1.count() << " and for existing KD: " << duration2.count() << "(ratio: " << duration2.count() / duration1.count() << ")" << std::endl;

    return std::tuple<double, double, double, double>{contour_advanced, contour, duration1.count(), duration2.count()};
}

void writeVectorsToFile(const std::vector<double>& vec1, const std::vector<double>& vec2, const std::vector<double>& vec3, const std::vector<double>& vec4, const std::vector<double>& vec5, const std::vector<std::vector<double>> mat, const std::string& filename) {
    if (vec1.size() != vec2.size() || vec1.size() != vec3.size()){
        std::cerr << "Error: Vectors must have the same size." << std::endl;
        return;
    }

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        file << vec1[i] << "\t" << vec2[i] << "\t" << vec3[i] << "\t" << vec4[i] << "\t" << vec5[i]; // Tab-separated columns
        for(double v : mat[i]){
            file << "\t" << v;
        }
        file << "\n";
    }

    file.close();
    std::cout << "Data written to " << filename << std::endl;
}

double log_uniform(double min, double max, std::mt19937& gen) {
    std::uniform_real_distribution<double> distrib(std::log10(min), std::log10(max));
    return std::pow(10, distrib(gen));
}

/*double GetKDa(double A, double scale){ return 1./(1.+EXP(-A/scale)); }
double GetKDb(double B, double scale){ return 1.-EXP(-B/scale); }
double GetKDA(double a, double scale){ 
    if(a<=0. || a>=1.)
        return Sign(a)*1.0e+300; 
    else return -scale*LOG(1./a-1.); 
}
double GetKDB(double b, double scale){ 
    if(b>=1.) 
        return 1.0e+300; 
    else if(b<=0.) 
        return 0.; 
    else return -scale*LOG(1.-b); 
}*/

int run_range(int fnum, double scale, double amin, double amax, double bmin, double bmax){

    int points = 100;
    int NumSamples = points/100;
    double BoundarySamplingFreq = 30.;//30.;//the larger the more points near the interval borders [0,1] of a & b
    //END USER INPUT

    cout << "testKD:" << endl;

    double RELERR = 0.1, ABSERR;
    double a,b,A,B,res,pmin,pmax;
    double MP = 1e-15;
    int D = 1;
    string FileNameSuffix = "results/Martin_Oscillations/";
    if (amin == 0.0 && amax == 1.0)
        FileNameSuffix +=  + "K" + to_string(D) + "_b/";
    else if(bmin == 0.0 && bmax == 1.0)
        FileNameSuffix +=  + "K" + to_string(D) + "_a/";
    else
        FileNameSuffix +=  + "K" + to_string(D) + "_zoom/";


    //double contour_advanced = 1;
    double contour, contour_advanced;
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000);
    double p, result;
    std::vector<double> results_KD;
    std::vector<double> results_contour;
    std::vector<double> a_values;
    std::vector<double> b_values;
    std::vector<double> A_values;
    std::vector<double> B_values;

    double gamma, eta;
    params_t params;
    double err=0.;
    struct KDintegrationParams KDip;




    if(bmax-bmin<MP){ // b=const
        cout << "@ b = " << bmin << " <-> B = " << GetKDB(bmin,scale) << endl;
        pmin = amin + 0.5 * (amax-amin) * ( 1. + tanh(BoundarySamplingFreq*(1./points - 0.5)) );
        pmax = amin + 0.5 * (amax-amin) * ( 1. + tanh(BoundarySamplingFreq*(0.5)) );
        cout << "        amin = " << pmin << " <-> minA = " << GetKDA(pmin,scale) << endl;
        cout << "        amax = " << pmax << " <-> maxA = " << GetKDA(pmax,scale) << endl;
        FileNameSuffix += to_string(fnum) + "_" + to_string(D) + "_b=" + to_string_with_precision(bmin,14);
        FileNameSuffix += "_amin=" + to_string_with_precision(amin,14) + ".txt";
        std::ofstream file(FileNameSuffix);
        file << std::fixed << std::setprecision(12);

        for (int i=1; i<=points; i++){

            p = amin + 0.5 * (amax-amin) * ( 1. + tanh(BoundarySamplingFreq*((double)i/points - 0.5)) );
            A = GetKDA(p,scale);
            B = GetKDB(bmin,scale);
            cout << "\n" << endl;
            cout << "a: " << p << " A: " << A << endl;
            cout << "b: " << bmin << " B: " << B << endl;
            if (A > 0 && B < 1e-2)
                gamma = 0.5 * 1./sqrt(A);
            else
                gamma = sqrt( (-A + sqrt(pow(A, 2) + pow(B, 3))) / (2*pow(B, 3)) );
            eta = A*gamma + 1./3. * pow(gamma, 3)*pow(B, 3) + 1./(4*gamma);
            params = {A, B, D, gamma, eta, 1};

            a_values.push_back(p);
            b_values.push_back(bmin);
            A_values.push_back(A);
            B_values.push_back(B);
            cout << "Running original function" << endl;
            contour = KD(D, A, B, err, 0.1, KDip);
            cout << "Running new function" << endl;
            contour_advanced = KD_contour(D, A, B, eta, gamma, workspace);
            //if (abs(contour_advanced)/abs(contour) > 1.1 || abs(contour_advanced)/abs(contour) < 0.9)
            //    contour_advanced = 0.;
            results_KD.push_back(contour);
            results_contour.push_back(contour_advanced);
            cout << "contour: " << contour << " contour_advanced: " << contour_advanced << endl;

        }

        for (size_t i = 0; i < a_values.size(); ++i) {
            file << a_values[i] << "\t" << b_values[i] << "\t" << A_values[i] << "\t" << B_values[i] << "\t" << results_KD[i] << "\t" << results_contour[i] << "\n"; // Tab-separated columns
        }
        file.close();

    }
    else{//a=const
        cout << "@ a = " << amin << " <-> A = " << GetKDA(amin,scale) << endl;
        pmin = bmin + 0.5 * (bmax-bmin) * ( 1. + tanh(BoundarySamplingFreq*(0.001 - 0.5)) );
        pmax = bmin + 0.5 * (bmax-bmin) * ( 1. + tanh(BoundarySamplingFreq*(0.5)) );
        cout << "        bmin = " << pmin << " <-> minB = " << GetKDB(pmin,scale) << endl;
        cout << "        bmax = " << pmax << " <-> maxB = " << GetKDB(pmax,scale) << endl;
        FileNameSuffix += to_string(fnum) + "_" + to_string(D) + "_a=" + to_string_with_precision(amin,14);
        FileNameSuffix += "_bmin=" + to_string_with_precision(bmin,14) + ".txt";

        std::ofstream file(FileNameSuffix);
        file << std::fixed << std::setprecision(12);

        for (int i=1; i<=points; i++){

            p = bmin + 0.5 * (bmax-bmin) * ( 1. + tanh(BoundarySamplingFreq*((double)i/points - 0.5)));
            A = GetKDA(amin,scale);
            B = GetKDB(p,scale);
            cout << "\n" << endl;
            cout << "a: " << amin << " A: " << A << endl;
            cout << "b: " << p << " B: " << B << endl;
            if (A > 0 && B < 1e-2)
                gamma = 0.5 * 1./sqrt(A);
            else
                gamma = sqrt( (-A + sqrt(pow(A, 2) + pow(B, 3))) / (2*pow(B, 3)) );
            eta = A*gamma + 1./3. * pow(gamma, 3)*pow(B, 3) + 1./(4*gamma);
            params = {A, B, D, gamma, eta, 1};
            //cout << "gamma:" << gamma << " eta:" << eta << " Contour value:" << KD_contour(D, A, B, eta, gamma, workspace) << endl;
            a_values.push_back(amin);
            b_values.push_back(p);
            A_values.push_back(A);
            B_values.push_back(B);
            cout << "Running original function" << endl;
            contour = KD(D, A, B, err, 0.1, KDip);
            cout << "Running new function" << endl;
            contour_advanced = KD_contour(D, A, B, eta, gamma, workspace);
            //if (abs(contour_advanced)/abs(contour) > 1.1 || abs(contour_advanced)/abs(contour) < 0.9)
            //    contour_advanced = 0.;
            results_KD.push_back(contour);
            results_contour.push_back(contour_advanced);
            cout << "contour: " << contour << " contour_advanced: " << contour_advanced << endl;

        }

        for (size_t i = 0; i < a_values.size(); ++i) {
            file << a_values[i] << "\t" << b_values[i] << "\t" << A_values[i] << "\t" << B_values[i] << "\t" << results_KD[i] << "\t" << results_contour[i] << "\n"; // Tab-separated columns
        }
        file.close();

    }



    gsl_integration_workspace_free(workspace);


    return 0;
}

int run_full_range(){
    double scale = 1.0e+6;

    std::vector<std::vector<double>> configurations = {

        /*{GetKDa(-1.e7,scale), GetKDa(-1.e7,scale), 0., 0.001}, 
        {GetKDa(-1.e6,scale), GetKDa(-1.e6,scale), 0., 0.001},
        {GetKDa(-1.e5,scale), GetKDa(-1.e5,scale), 0., 0.001},     
        {GetKDa(0.,scale), GetKDa(0.,scale), 0., 0.001},
        {GetKDa(1.e5,scale), GetKDa(1.e5,scale), 0., 0.001}, 
        {GetKDa(1.e6,scale), GetKDa(1.e6,scale), 0., 0.001},
        {GetKDa(1.e7,scale), GetKDa(1.e7,scale), 0., 0.001},
        
        {GetKDa(-1.e7,scale), GetKDa(-1.e7,scale), 0.5, 0.505}, 
        {GetKDa(-1.e6,scale), GetKDa(-1.e6,scale), 0.5, 0.505},
        {GetKDa(-1.e5,scale), GetKDa(-1.e5,scale), 0.5, 0.505},     
        {GetKDa(0.,scale), GetKDa(0.,scale), 0.5, 0.505},
        {GetKDa(1.e5,scale), GetKDa(1.e5,scale), 0.5, 0.505}, 
        {GetKDa(1.e6,scale), GetKDa(1.e6,scale), 0.5, 0.505},
        {GetKDa(1.e7,scale), GetKDa(1.e7,scale), 0.5, 0.505}, 

        {GetKDa(-1.e7,scale), GetKDa(-1.e7,scale), 0.9999, 0.999901}, 
        {GetKDa(-1.e6,scale), GetKDa(-1.e6,scale), 0.9999, 0.999901},
        {GetKDa(-1.e5,scale), GetKDa(-1.e5,scale), 0.9999, 0.999901},     
        {GetKDa(0.,scale), GetKDa(0.,scale), 0.9999, 0.999901},
        {GetKDa(1.e5,scale), GetKDa(1.e5,scale), 0.9999, 0.999901}, 
        {GetKDa(1.e6,scale), GetKDa(1.e6,scale), 0.9999, 0.999901},
        {GetKDa(1.e7,scale), GetKDa(1.e7,scale), 0.9999, 0.999901}, 

        {GetKDa(1.0e1,scale), GetKDa(1.0e1,scale), 0., 0.001}, 
        {GetKDa(1.0e1,scale), GetKDa(1.0e1,scale), 0.5, 0.9999},
        {GetKDa(1.0e1,scale), GetKDa(1.0e1,scale), 0.505, 0.999901},

        //{0.495, 0.505, GetKDb(1.e-4,scale), GetKDb(1.e-4,scale)}, 
        //{0.495, 0.505, GetKDb(1.e-3,scale), GetKDb(1.e-3,scale)}, 
        //{0.495, 0.505, GetKDb(1.e-2,scale), GetKDb(1.e-2,scale)}, 
        //{0.495, 0.505, GetKDb(1.e-1,scale), GetKDb(1.e-1,scale)}, 
        {0.495, 0.505, GetKDb(1.e0,scale), GetKDb(1.e0,scale)}, 
        {0.495, 0.505, GetKDb(1.e1,scale), GetKDb(1.e1,scale)}, 
        {0.495, 0.505, GetKDb(1.e2,scale), GetKDb(1.e2,scale)}, 
        {0.495, 0.505, GetKDb(1.e3,scale), GetKDb(1.e3,scale)}, 
        {0.495, 0.505, GetKDb(1.e4,scale), GetKDb(1.e4,scale)}, 
        {0.495, 0.505, GetKDb(1.e5,scale), GetKDb(1.e5,scale)}, 
        {0.495, 0.505, GetKDb(1.e6,scale), GetKDb(1.e6,scale)}, 
        {0.495, 0.505, GetKDb(1.e7,scale), GetKDb(1.e7,scale)}, 

        //{0.498, 0.508, GetKDb(1.e-4,scale), GetKDb(1.e-4,scale)}, 
        //{0.498, 0.508, GetKDb(1.e-3,scale), GetKDb(1.e-3,scale)}, 
        //{0.498, 0.508, GetKDb(1.e-2,scale), GetKDb(1.e-2,scale)}, 
        //{0.498, 0.508, GetKDb(1.e-1,scale), GetKDb(1.e-1,scale)}, 
        {0.498, 0.508, GetKDb(1.e0,scale), GetKDb(1.e0,scale)}, 
        {0.498, 0.508, GetKDb(1.e1,scale), GetKDb(1.e1,scale)}, 
        {0.498, 0.508, GetKDb(1.e2,scale), GetKDb(1.e2,scale)}, 
        {0.498, 0.508, GetKDb(1.e3,scale), GetKDb(1.e3,scale)}, 
        {0.498, 0.508, GetKDb(1.e4,scale), GetKDb(1.e4,scale)}, 
        {0.498, 0.508, GetKDb(1.e5,scale), GetKDb(1.e5,scale)}, 
        {0.498, 0.508, GetKDb(1.e6,scale), GetKDb(1.e6,scale)}, 
        {0.498, 0.508, GetKDb(1.e7,scale), GetKDb(1.e7,scale)}, 

        {GetKDa(3.0e+4,scale), GetKDa(3.0e+4,scale), 0.5, 0.505}*/

        /*{0, 4.5397868702434e-05, 4.5397868702434e-05, 0.0, 1.0},
        {1, 0.26894142137, 0.26894142137, 0.0, 1.0},
        {2, 0.47502081252106, 0.47502081252106, 0.0, 1.0},
        {3, 0.500001, 0.500001, 0.0, 1.0},
        {4, 0.52497918747894, 0.52497918747894, 0.0, 1.0},
        {5, 0.73105857863, 0.73105857863, 0.0, 1.0},
        {6, 0.9999546021313, 0.9999546021313, 0.0, 1.0},

        {0, 0.0, 1.0, 1.0000000827404e-10, 1.0000000827404e-10},
        {1, 0.0, 1.0, 9.9999997171807e-10, 9.9999997171807e-10},
        {2, 0.0, 1.0, 9.9999999392253e-09, 9.9999999392253e-09},
        {3, 0.0, 1.0, 9.9999994951361e-08, 9.9999994951361e-08},
        {4, 0.0, 1.0, 9.9999949998431e-07, 9.9999949998431e-07},
        {5, 0.0, 1.0, 9.9999500001724e-06, 9.9999500001724e-06},
        {6, 0.0, 1.0, 9.9995000166664e-05, 9.9995000166664e-05},
        {7, 0.0, 1.0, 0.00099950016662498, 0.00099950016662498},
        {8, 0.0, 1.0, 0.0099501662508319, 0.0099501662508319},
        {9, 0.0, 1.0, 0.09516258196404, 0.09516258196404},
        {10, 0.0, 1.0, 0.63212055882856, 0.63212055882856},
        {11, 0.0, 1.0, 0.99995460007024, 0.99995460007024}, */

        {0, 0.51, 0.51, 0.0, 0.001}, // disconnected contours
        /*{1, 0.5, 0.5, 0.5, 0.505},
        {2, 0.5, 0.5, 0.9999, 0.999901},// ?
        {3, 0.5000025, 0.5000025, 0.0, 0.001},
        {4, 0.5000025, 0.5000025, 0.5, 0.505},
        {5, 0.5000025, 0.5000025, 0.9999, 0.999901}, //?
        {6, 0.50749943755062, 0.50749943755062, 0.5, 0.505},
        {7, 0.498, 0.508, 9.9999949998431e-07, 9.9999949998431e-07},
        {8, 0.9999, 0.99995, 9.9999949998431e-07, 9.9999949998431e-07},
        {9, 0.496, 0.506, 1.0000000827404e-10, 1.0000000827404e-10}*/


    };
    
    // Loop through each configuration
    for (const auto& config : configurations) {
        int fnum = config[0];
        double amin = config[1];
        double amax = config[2];
        double bmin = config[3];
        double bmax = config[4];
        
        // Call the function with the current set of parameters
        run_range(fnum, scale, amin, amax, bmin, bmax);
    }
    return 0;
}

int main_local(){

    std::vector<double> B_values;
    std::vector<double> contour_values;
    std::vector<double> contour_advanced_values;
    std::vector<double> duration1;
    std::vector<double> duration2;
    std::vector<std::vector<double>> matrix;
    std::vector<std::vector<double>> matrix1;
    std::vector<std::vector<double>> matrix2;

    std::vector<double> fc;
    std::random_device rd;  // Obtain a random seed
    std::mt19937 gen(rd()); // Mersenne Twister PRNG
    double min_val = 1.0e4;   // Smallest possible value
    double max_val = 1.0e7; // Largest possible value

    // Generate a log-uniform random number
    auto [ca, c, d1, d2] = run_contour(1, 0, 0.003372003904, fc);
    //return 0;


    run_full_range();
    return 0;

    double A1, B1;
    for(int i=0; i<100; i++){
        A1=log_uniform(min_val, max_val, gen);
        B1=log_uniform(min_val, max_val, gen);
        auto [ca, c, d1, d2] = run_contour(3, A1, B1, fc);
        fc.clear();
    }
    return 0;
    A1 = 29217.5;
    B1 = 2.35441e+06;
    run_contour(3, A1, B1, fc);
    for(double v: fc){
        cout << v << " ";
    }

    int D=3;
    double A = 1;
    //std::cout << "Time taken for contour:" << std::endl;
    fc.clear();
    for(double B=0.1; B<10; B+=0.1){
        B_values.push_back(B);
        auto [ca, c, d1, d2] = run_contour(D, A, B, fc);
        contour_advanced_values.push_back(ca);
        contour_values.push_back(c);
        duration1.push_back(d1);
        duration2.push_back(d2);
        matrix.push_back(fc);
        fc.clear();
    }

    writeVectorsToFile(B_values, contour_advanced_values, contour_values, duration1, duration2, matrix, "results/A_1_B_small.txt");
    std::cout << "Time taken for contour:" << std::endl;

    B_values.clear();
    contour_values.clear();
    contour_advanced_values.clear();

    for(double B=1; B<1000; B+=10){
        B_values.push_back(B);
        auto [ca, c, d1, d2] = run_contour(D, A, B, fc);
        contour_advanced_values.push_back(ca);
        contour_values.push_back(c);
        duration1.push_back(d1);
        duration2.push_back(d2);
        matrix1.push_back(fc);
        fc.clear();
    }
    fc.clear();
    writeVectorsToFile(B_values, contour_advanced_values, contour_values, duration1, duration2, matrix1, "results/A_1_B_large.txt");


    B_values.clear();
    contour_values.clear();
    contour_advanced_values.clear();
    
    double B=1;
    for(double Av=-1000; Av<1000; Av+=10){
        B_values.push_back(Av);
        auto [ca, c, d1, d2] = run_contour(D, Av, B, fc);
        contour_advanced_values.push_back(ca);
        contour_values.push_back(c);
        duration1.push_back(d1);
        duration2.push_back(d2);
        matrix2.push_back(fc);
        fc.clear();
    }
    fc.clear();
    writeVectorsToFile(B_values, contour_advanced_values, contour_values, duration1, duration2, matrix2, "results/A_large_B_1.txt");

    return 0;
}

//END Michael Tsesmelis code
