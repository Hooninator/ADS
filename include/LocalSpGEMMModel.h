
#ifndef LOCALSPGEMMMODEL_H
#define LOCALSPGEMMMODEL_H

#include "common.h"
#include "SpParMatInfo.h"
#include "PlatformParams.h"

namespace autotuning {
using namespace combblas;

enum FLOPS_STRAT {
    FLOPS_GLOB_DENSITY,
    FLOPS_LOC_DENSITY,
    FLOPS_NNZ
};

template <typename AIT, typename BIT>
struct LocalSpGEMMInfo {

    
    long long FLOPS;

    AIT rowsA;
    AIT colsA;
    BIT rowsB;
    BIT colsB;
    AIT nnzA;
    BIT nnzB;

    float globDensityA;
    float locDensityA;
    float globDensityB;
    float locDensityB;

    void SetFLOPS(SpGEMMParams& params, FLOPS_STRAT strat) {
        switch(strat) {
            case FLOPS_GLOB_DENSITY:
                SetFLOPSGlobalDensity(params);
                break;
            case FLOPS_LOC_DENSITY:
                SetFLOPSLocalDensity(params);
                break;
            case FLOPS_NNZ:
                SetFLOPSNnz(params);
                break;
            default:
                UNREACH_ERR();
        }
    }

    /* Approximate local FLOPS using global density-based nnz estimation */
    void SetFLOPSGlobalDensity(SpGEMMParams& params) {

        long long tileFLOPS = globDensityA * (rowsA) * // estimate nnz per col of A
                        globDensityB * rowsB * // estimate nnz per col of B
                        colsB ; // once per col of B

#ifdef PROFILE
#endif

        FLOPS = tileFLOPS;

    }

    /* Approximate local FLOPS using global density-based nnz estimation */
    void SetFLOPSLocalDensity(SpGEMMParams& params) {

        long long tileFLOPS = locDensityA * (rowsA) * // estimate nnz per col of A
                        locDensityB * rowsB * // estimate nnz per col of B
                        colsB ; // once per col of B

#ifdef PROFILE
#endif

        FLOPS = tileFLOPS;

    }


    /* Approximate local FLOPS using  nnzA and nnzB */
    void SetFLOPSNnz(SpGEMMParams& params) {

        long long tileFLOPS = nnzA*nnzB; 
                            
#ifdef PROFILE
#endif

        FLOPS = tileFLOPS;

    }

};


template <typename AIT, typename BIT>
class LocalSpGEMMModel {

public:
    LocalSpGEMMModel(){}
    
    virtual double Time(LocalSpGEMMInfo<AIT, BIT> * info) {INVALID_CALL_ERR();}

};


template <typename AIT, typename ANT, typename BIT, typename BNT>
class RooflineLocalSpGEMMModel: public LocalSpGEMMModel<AIT, BIT> {
public:
    RooflineLocalSpGEMMModel(PlatformParams& params):
        params(params)
    {
        
    }


    double Time(LocalSpGEMMInfo<AIT, BIT> * info) {

        ASSERT(info->FLOPS>-1, "FLOPS for localSpGEMM should be a positive number, but got " + std::to_string(info->FLOPS));

        AIT bytesReadA = info->nnzA*sizeof(ANT) + info->nnzA*sizeof(AIT) + info->nnzA*sizeof(AIT); 
        BIT bytesReadB = info->nnzB*sizeof(BNT) + info->nnzB*sizeof(BIT) + info->nnzB*sizeof(BIT); 

        AIT totalBytes = bytesReadA + bytesReadB; //TODO: cast as whichever type is larger

        double memMovementTime = totalBytes/params.GetMemBW(); // memBW is MB/s==B/us
        double computationTime = info->FLOPS * params.GetCostFLOP() + std::log2(info->nnzB);  //heap cost
                                                                                              
        //TODO: What about hashSpGEMM?

#ifdef PROFILE
#endif

        return memMovementTime + computationTime;

    }


    inline PlatformParams GetParams() {return params;}

private:
    PlatformParams params;
};


/* T = P(x), P(x) = \sum_{i=0}^d(a*x^i) 
template <typename IT>
class RegressionLocalSpGEMMModel : public LocalSpGEMMModel<IT> {
    
public:

    RegressionLocalSpGEMMModel(std::vector<double>& coeffs):
    coeffs(coeffs) 
    {
        
    }
    
    double Time(LocalSpGEMMInfo<IT> * info) {

        int i=0;

        double timeSeconds = std::accumulate(coeffs.begin(), coeffs.end(), 0.0, [=](double sum, double a)mutable {
            return sum + a*std::pow(info->FLOPS,i++);
        });
        
        return timeSeconds * 1e6; //convert to us

    }
    
    inline int Degree() const {return coeffs.size();}

private:
    
    std::vector<double> coeffs;

};

//TODO: Multivariate Regression Model

//computed using numpy least squares fitting functionality
//TODO: C++ program to do this
//This is going to make things complicated...
std::vector<double> regSpGEMMPerlmutter {1.37959216, 3.96351051e-07};
*/

}//combblas



#endif


